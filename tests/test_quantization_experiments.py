import ast
import importlib
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _function_source(path: str, function_name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{function_name} not found in {path}")


def _method_source(path: str, class_name: str, method_name: str) -> str:
    source = _source(path)
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return ast.get_source_segment(source, item)
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


class QuantizationExperimentTests(unittest.TestCase):
    def test_class_balanced_indices_are_deterministic_and_balanced(self):
        utils = importlib.import_module("my_optim.experiment_utils")
        targets = [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5

        first = utils.class_balanced_indices(targets, sample_size=8, seed=17)
        second = utils.class_balanced_indices(targets, sample_size=8, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(len(set(first)), 8)
        self.assertEqual(Counter(targets[index] for index in first), {0: 2, 1: 2, 2: 2, 3: 2})

    def test_class_balanced_indices_distribute_remainder_without_duplicates(self):
        utils = importlib.import_module("my_optim.experiment_utils")
        targets = [0] * 4 + [1] * 4 + [2] * 4

        indices = utils.class_balanced_indices(targets, sample_size=8, seed=3)
        counts = Counter(targets[index] for index in indices)

        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(len(indices), 8)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_modelopt_exclusion_rules_disable_both_quantizers_in_order(self):
        utils = importlib.import_module("my_optim.experiment_utils")

        rules = utils.make_modelopt_exclusion_rules(["mlp.fc1", "attn.qkv"])

        self.assertEqual(list(rules), [
            "*mlp.fc1*weight_quantizer",
            "*mlp.fc1*input_quantizer",
            "*attn.qkv*weight_quantizer",
            "*attn.qkv*input_quantizer",
        ])
        self.assertTrue(all(rule == {"enable": False} for rule in rules.values()))

    def test_trt_inference_wrappers_do_not_synchronize_inside_call(self):
        for path in ("my_optim/run_trt_eval.py", "my_optim_Trans/run_trt_eval.py"):
            method = _method_source(path, "TRTInferencer", "__call__")
            self.assertNotIn("torch.cuda.synchronize()", method)

        for path in ("my_optim/run_summary.py", "my_optim_Trans/run_summary.py"):
            function = _function_source(path, "load_trt_infer")
            self.assertNotIn("torch.cuda.synchronize()", function)

    def test_accuracy_evaluation_synchronizes_before_reading_logits(self):
        for path in ("my_optim/common.py", "my_optim_Trans/common.py"):
            function = _function_source(path, "evaluate_accuracy")
            self.assertIn("torch.cuda.synchronize()", function)

    def test_fallback_uses_balanced_subset_and_supported_quant_cfg_rules(self):
        source = _source("my_optim_Trans/run_sensitive_fallback.py")

        self.assertIn("class_balanced_indices", source)
        self.assertIn("SubsetRandomSampler", source)
        self.assertIn("make_modelopt_exclusion_rules", source)
        self.assertNotIn('"override_fn"', source)
        self.assertIn('"fallback_search.json"', source)

    def test_cnn_builder_uses_corrected_int8_and_tactic_settings(self):
        source = _source("my_optim/build_trt_engine.py")

        self.assertIn("trt.BuilderFlag.FP16", source)
        self.assertIn("trt.BuilderFlag.INT8", source)
        self.assertIn("builder_optimization_level = 5", source)
        self.assertIn("avg_timing_iterations", source)
        self.assertIn("create_timing_cache", source)
        self.assertIn("--opt-bs", source)
        self.assertIn("--max-bs", source)

    def test_transformer_eval_can_load_mixed_precision_engine(self):
        source = _source("my_optim_Trans/run_trt_eval.py")

        self.assertIn("--engine-suffix", source)
        self.assertIn("--result-tag", source)

    def test_accuracy_constrained_search_prefers_smallest_acceptable_fallback(self):
        utils = importlib.import_module("my_optim.experiment_utils")
        trials = [
            {"name": "full_int8", "top1": 60.0, "fallback_cost": 0},
            {"name": "wide_fallback", "top1": 75.7, "fallback_cost": 6},
            {"name": "small_fallback", "top1": 75.4, "fallback_cost": 2},
        ]

        selected = utils.select_accuracy_constrained_candidate(
            trials,
            fp32_top1=75.8,
            target_drop=0.5,
        )

        self.assertEqual(selected["name"], "small_fallback")

    def test_cnn_search_uses_architecture_aware_mobile_candidates(self):
        source = _source("my_optim/run_sensitive_fallback.py")

        self.assertIn("mobilenetv3_large_100", source)
        self.assertIn('"conv_dw"', source)
        self.assertIn('"se.conv_reduce"', source)
        self.assertIn('"blocks.0"', source)
        self.assertIn('"blocks.5"', source)
        self.assertIn("class_balanced_indices", source)
        self.assertIn("select_accuracy_constrained_candidate", source)
        self.assertIn('"int8_qdq_search_selected_inline.onnx"', source)

    def test_cnn_eval_can_load_search_engine_and_use_bs32_throughput(self):
        source = _source("my_optim/run_trt_eval.py")

        self.assertIn("--engine-suffix", source)
        self.assertIn("--result-tag", source)
        self.assertIn("--throughput-batch-size", source)


if __name__ == "__main__":
    unittest.main()
