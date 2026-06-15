"""Search MobileNetV3 activation INT8 scales without changing layer precision."""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).parent.parent))

import common
from my_optim.scale_search_core import run_scale_search


MOBILENET_SCALE_GROUPS = [
    {"name": "stem", "fragments": ["conv_stem"]},
    {"name": "depthwise", "fragments": ["conv_dw"]},
    {
        "name": "squeeze_excite",
        "fragments": ["se.conv_reduce", "se.conv_expand"],
    },
    {"name": "pointwise", "fragments": ["conv_pw", "conv_pwl"]},
    {"name": "head", "fragments": ["conv_head", "classifier"]},
]


if __name__ == "__main__":
    run_scale_search(
        common=common,
        groups=MOBILENET_SCALE_GROUPS,
        expected_model_prefixes=["mobilenetv3_large_100"],
        description="MobileNetV3 ModelOpt activation amax/scale search",
    )
