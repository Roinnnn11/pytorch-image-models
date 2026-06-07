"""
Quick smoke-test: forward pass for CNN and Transformer models.
Usage: python run_models_check.py
"""
import torch
import timm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 2
IMG_SZ = 224

MODELS = {
    # CNN
    "resnet50":                    {"input_size": (3, IMG_SZ, IMG_SZ)},
    "resnet18":                    {"input_size": (3, IMG_SZ, IMG_SZ)},
    "mobilenetv3_large_100":       {"input_size": (3, IMG_SZ, IMG_SZ)},
    "mobilenetv2_100":             {"input_size": (3, IMG_SZ, IMG_SZ)},
    # Transformer
    "vit_base_patch16_224":        {"input_size": (3, IMG_SZ, IMG_SZ)},
    "vit_small_patch16_224":       {"input_size": (3, IMG_SZ, IMG_SZ)},
    "swin_base_patch4_window7_224":{"input_size": (3, IMG_SZ, IMG_SZ)},
    "swin_tiny_patch4_window7_224":{"input_size": (3, IMG_SZ, IMG_SZ)},
}

def run(name, cfg):
    x = torch.randn(BATCH, *cfg["input_size"]).to(DEVICE)
    model = timm.create_model(name, pretrained=False).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(x)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return out.shape, params

if __name__ == "__main__":
    print(f"Device: {DEVICE}  |  timm {timm.__version__}  |  torch {torch.__version__}\n")
    print(f"{'Model':<40} {'Output shape':<20} {'Params (M)'}")
    print("-" * 70)
    ok, fail = 0, 0
    for name, cfg in MODELS.items():
        try:
            shape, params = run(name, cfg)
            print(f"{'[OK] ' + name:<40} {str(shape):<20} {params:.1f}M")
            ok += 1
        except Exception as e:
            print(f"{'[FAIL] ' + name:<40} ERROR: {e}")
            fail += 1
    print("-" * 70)
    print(f"\nResult: {ok} passed, {fail} failed")
