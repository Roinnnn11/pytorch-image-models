"""Search ViT activation INT8 scales without changing layer precision."""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))

import common
from my_optim.scale_search_core import run_scale_search


VIT_SCALE_GROUPS = [
    {"name": "patch_embed", "fragments": ["patch_embed.proj"]},
    {"name": "attention_qkv", "fragments": ["attn.qkv"]},
    {"name": "attention_proj", "fragments": ["attn.proj"]},
    {"name": "mlp_fc1", "fragments": ["mlp.fc1"]},
    {"name": "mlp_fc2", "fragments": ["mlp.fc2"]},
    {"name": "head", "fragments": ["head"]},
]


if __name__ == "__main__":
    run_scale_search(
        common=common,
        groups=VIT_SCALE_GROUPS,
        expected_model_prefixes=["vit_base_patch16_224"],
        description="ViT ModelOpt activation amax/scale search",
    )
