"""Build, evaluate, and summarize a fixed-batch ViT INT8 experiment matrix."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from my_optim.experiment_utils import pareto_front, parse_fixed_batches


MODEL = "vit_base_patch16_224"
DEFAULT_BATCHES = "1,4,16,32"
DEFAULT_SQ_ALPHAS = "0.3,0.5,0.7,0.9"
BASE = Path(__file__).resolve().parent
ONNX_DIR = BASE / "onnx" / MODEL
ENGINE_DIR = BASE / "engines" / MODEL
RESULTS_DIR = BASE / "results" / MODEL


def parse_alphas(value: str) -> list[float]:
    try:
        alphas = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("SmoothQuant alphas must be comma-separated numbers") from error
    if not alphas or any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("SmoothQuant alphas must be in [0, 1]")
    if len(set(alphas)) != len(alphas):
        raise ValueError("SmoothQuant alphas must be unique")
    return alphas


def alpha_tag(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def candidate_specs(alphas: list[float]) -> list[dict]:
    specs = [
        {"name": "max", "ptq": ["--calib", "max"], "onnx": "int8_qdq_max_inline.onnx"},
        {"name": "mse", "ptq": ["--calib", "mse"], "onnx": "int8_qdq_mse_inline.onnx"},
    ]
    for alpha in alphas:
        name = f"smoothquant_a{alpha_tag(alpha)}"
        specs.append({
            "name": name,
            "ptq": ["--calib", "smoothquant", "--smoothquant-alpha", f"{alpha:g}"],
            "onnx": f"int8_qdq_{name}_inline.onnx",
        })
    specs.extend([
        {"name": "scale_search", "ptq": None, "onnx": "int8_qdq_scale_search_inline.onnx"},
        {"name": "mlp_fallback", "ptq": None, "onnx": "int8_qdq_mixed_inline.onnx"},
    ])
    return specs


def artifact_path(directory: Path, suffix: str) -> Path:
    return directory / f"{MODEL}_{suffix}"


def run_command(command: list[str], output: Path, force: bool, dry_run: bool) -> bool:
    if output.exists() and not force:
        print(f"[reuse] {output}")
        return True
    print("[run] " + " ".join(command))
    if dry_run:
        return True
    completed = subprocess.run(command, cwd=BASE, check=False)
    if completed.returncode != 0:
        print(f"[failed:{completed.returncode}] {' '.join(command)}", file=sys.stderr)
        return False
    if not output.exists():
        print(f"[failed] command succeeded but artifact is missing: {output}", file=sys.stderr)
        return False
    return True


def extract_fp32_top1(path: Path) -> float:
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("fp32", "pytorch_fp32"):
        if isinstance(data.get(key), dict) and "top1" in data[key]:
            return float(data[key]["top1"])
    if "top1" in data:
        return float(data["top1"])
    raise ValueError(f"cannot find FP32 top1 in {path}")


def load_trials(batches: list[int]) -> list[dict]:
    trials = []
    for result_path in sorted(RESULTS_DIR.glob("trt_int8_*_bs*.json")):
        data = json.loads(result_path.read_text(encoding="utf-8"))
        batch = data.get("fixed_batch_size")
        if batch not in batches:
            continue
        latency = data.get("latency", {}).get(f"bs{batch}", {})
        if not latency or "top1" not in data:
            continue
        tag = str(data.get("tag", result_path.stem))
        marker = "trt_int8_"
        candidate = tag[len(marker):].rsplit(f"_bs{batch}", 1)[0] if tag.startswith(marker) else tag
        trials.append({
            "name": candidate,
            "batch_size": int(batch),
            "top1": float(data["top1"]),
            "top5": float(data["top5"]),
            "evaluated_samples": int(data.get("evaluated_samples", data.get("n", 0))),
            "latency_ms": float(latency["latency_ms_mean"]),
            "throughput": float(latency["throughput_img_s"]),
            "e2e_throughput": float(data.get("e2e_throughput_img_s", 0.0)),
            "cuda_graph_enabled": bool(data.get("cuda_graph_enabled", False)),
            "result": str(result_path),
        })
    return trials


def write_summary(trials: list[dict], fp32_top1: float, max_drop: float, batches: list[int]) -> None:
    fronts = {}
    for batch in batches:
        batch_trials = [trial for trial in trials if trial["batch_size"] == batch]
        fronts[str(batch)] = pareto_front(batch_trials, fp32_top1, max_drop)
    payload = {
        "model": MODEL,
        "fp32_top1": fp32_top1,
        "max_top1_drop": max_drop,
        "trials": trials,
        "pareto_by_batch": fronts,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "vit_int8_matrix.json"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    json_tmp.replace(json_path)

    lines = [
        "# ViT INT8 fixed-batch matrix",
        "",
        f"FP32 Top-1: {fp32_top1:.3f}% · maximum drop: {max_drop:.3f} points",
        "",
        "| Batch | Candidate | Top-1 | Drop | GPU latency | GPU throughput | E2E throughput | Pareto |",
        "|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    pareto_keys = {
        (trial["name"], trial["batch_size"])
        for front in fronts.values()
        for trial in front
    }
    for trial in sorted(trials, key=lambda item: (item["batch_size"], -item["top1"], item["latency_ms"])):
        drop = fp32_top1 - trial["top1"]
        eligible = drop <= max_drop
        pareto = "yes" if (trial["name"], trial["batch_size"]) in pareto_keys else ""
        lines.append(
            f"| {trial['batch_size']} | {trial['name']} | {trial['top1']:.3f}% | "
            f"{drop:.3f} | {trial['latency_ms']:.3f} ms | {trial['throughput']:.0f} img/s | "
            f"{trial['e2e_throughput']:.0f} img/s | {pareto if eligible else 'ineligible'} |"
        )
    markdown_path = RESULTS_DIR / "vit_int8_matrix.md"
    markdown_tmp = markdown_path.with_suffix(".md.tmp")
    markdown_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    markdown_tmp.replace(markdown_path)
    print(f"summary: {json_path}")
    print(f"table: {markdown_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default=DEFAULT_BATCHES)
    parser.add_argument("--smoothquant-alphas", default=DEFAULT_SQ_ALPHAS)
    parser.add_argument("--candidates", default=None, help="Comma-separated candidate names")
    parser.add_argument("--calib-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-top1-drop", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_top1_drop < 0 or args.calib_samples <= 0:
        parser.error("accuracy drop must be non-negative and calibration size positive")
    try:
        batches = parse_fixed_batches(args.batches)
        alphas = parse_alphas(args.smoothquant_alphas)
    except ValueError as error:
        parser.error(str(error))

    specs = candidate_specs(alphas)
    if args.candidates:
        selected = {item.strip() for item in args.candidates.split(",") if item.strip()}
        known = {spec["name"] for spec in specs}
        unknown = selected - known
        if unknown:
            parser.error(f"unknown candidates: {', '.join(sorted(unknown))}")
        specs = [spec for spec in specs if spec["name"] in selected]

    env_model = os.environ.get("OPTIM_MODEL", MODEL)
    if env_model != MODEL:
        parser.error(f"OPTIM_MODEL must be {MODEL}, got {env_model}")
    os.environ["OPTIM_MODEL"] = MODEL
    python = sys.executable
    failures = []

    for spec in specs:
        onnx_path = artifact_path(ONNX_DIR, spec["onnx"])
        if spec["ptq"] is not None:
            ptq_command = [
                python,
                "run_ptq_int8.py",
                *spec["ptq"],
                "--calib-samples", str(args.calib_samples),
                "--seed", str(args.seed),
            ]
            if not run_command(ptq_command, onnx_path, args.force, args.dry_run):
                failures.append({"candidate": spec["name"], "stage": "ptq"})
                continue
        elif not onnx_path.exists():
            print(f"[skip] optional candidate missing: {onnx_path}")
            continue

        for batch in batches:
            engine_suffix = f"int8_{spec['name']}_bs{batch}.engine"
            engine_path = artifact_path(ENGINE_DIR, engine_suffix)
            build_command = [
                python,
                "build_trt_engine.py",
                "--precision", "int8",
                "--onnx-suffix", spec["onnx"],
                "--engine-suffix", engine_suffix,
                "--fixed-bs", str(batch),
            ]
            if not run_command(build_command, engine_path, args.force, args.dry_run):
                failures.append({"candidate": spec["name"], "batch": batch, "stage": "build"})
                continue

            result_tag = f"trt_int8_{spec['name']}_bs{batch}"
            result_path = RESULTS_DIR / f"{result_tag}.json"
            eval_command = [
                python,
                "run_trt_eval.py",
                "--precision", "int8",
                "--engine-suffix", engine_suffix,
                "--result-tag", result_tag,
                "--fixed-bs", str(batch),
                "--workers", str(args.workers),
            ]
            if args.use_cuda_graph:
                eval_command.append("--use-cuda-graph")
            if not run_command(eval_command, result_path, args.force, args.dry_run):
                failures.append({"candidate": spec["name"], "batch": batch, "stage": "eval"})

        consistency_path = RESULTS_DIR / f"consistency_{spec['name']}.json"
        consistency_command = [
            python,
            "run_trt_consistency.py",
            "--candidate", spec["name"],
            "--batches", ",".join(str(batch) for batch in batches),
        ]
        engines_ready = all(
            artifact_path(ENGINE_DIR, f"int8_{spec['name']}_bs{batch}.engine").exists()
            for batch in batches
        )
        if args.dry_run or engines_ready:
            if not run_command(
                consistency_command,
                consistency_path,
                args.force,
                args.dry_run,
            ):
                failures.append({"candidate": spec["name"], "stage": "consistency"})

    if args.dry_run:
        return
    baseline_path = RESULTS_DIR / "baseline.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"run run_baseline.py first; missing {baseline_path}")
    trials = load_trials(batches)
    if not trials:
        raise RuntimeError("no completed fixed-batch evaluation results were found")
    write_summary(trials, extract_fp32_top1(baseline_path), args.max_top1_drop, batches)
    if failures:
        failure_path = RESULTS_DIR / "vit_int8_matrix_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"completed with {len(failures)} failed stages: {failure_path}")


if __name__ == "__main__":
    main()
