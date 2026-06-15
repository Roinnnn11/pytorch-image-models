import importlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class ScaleSearchTests(unittest.TestCase):
    def test_scale_ratio_parser_rejects_invalid_values(self):
        utils = importlib.import_module("my_optim.scale_search_utils")

        self.assertEqual(utils.parse_scale_ratios("1.0,0.9,0.75"), [1.0, 0.9, 0.75])
        with self.assertRaises(ValueError):
            utils.parse_scale_ratios("1.0,0")
        with self.assertRaises(ValueError):
            utils.parse_scale_ratios("")

    def test_best_scale_prefers_accuracy_then_less_clipping(self):
        utils = importlib.import_module("my_optim.scale_search_utils")
        trials = [
            {"ratio": 0.7, "top1": 75.4, "top5": 92.0},
            {"ratio": 0.9, "top1": 75.4, "top5": 92.0},
            {"ratio": 0.8, "top1": 75.3, "top5": 92.5},
        ]

        selected = utils.select_best_scale_trial(trials)

        self.assertEqual(selected["ratio"], 0.9)

    def test_group_matching_is_fragment_based(self):
        utils = importlib.import_module("my_optim.scale_search_utils")

        self.assertTrue(
            utils.quantizer_matches(
                "blocks.3.1.conv_dw.input_quantizer",
                ["conv_dw", "se.conv_reduce"],
            )
        )
        self.assertFalse(
            utils.quantizer_matches(
                "blocks.3.1.conv_pw.input_quantizer",
                ["conv_dw", "se.conv_reduce"],
            )
        )

    def test_core_search_writes_activation_amax_and_exports_independent_artifacts(self):
        source = _source("my_optim/scale_search_core.py")

        self.assertIn("TensorQuantizer", source)
        self.assertIn('"input_quantizer"', source)
        self.assertIn("._amax", source)
        self.assertIn("/ 127.0", source)
        self.assertIn("class_balanced_indices", source)
        self.assertIn('"scale_search.json"', source)
        self.assertIn('"int8_qdq_scale_search.onnx"', source)
        self.assertIn('"int8_qdq_scale_search_inline.onnx"', source)

    def test_mobile_and_vit_define_architecture_specific_scale_groups(self):
        mobile = _source("my_optim/run_scale_search.py")
        vit = _source("my_optim_Trans/run_scale_search.py")

        for fragment in ("conv_stem", "conv_dw", "se.conv_reduce", "conv_pw"):
            self.assertIn(fragment, mobile)
        for fragment in ("patch_embed", "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            self.assertIn(fragment, vit)

    def test_new_one_click_scripts_do_not_replace_fallback_scripts(self):
        mobile = _source("my_optim/run_mobile_scale_search.sh")
        vit = _source("my_optim_Trans/run_vit_scale_search.sh")
        old_mobile = _source("my_optim/run_mobile_search.sh")

        self.assertIn("run_scale_search.py", mobile)
        self.assertIn("int8_qdq_scale_search_inline.onnx", mobile)
        self.assertIn("int8_scale_search.engine", mobile)
        self.assertIn("run_scale_search.py", vit)
        self.assertIn("int8_qdq_scale_search_inline.onnx", vit)
        self.assertIn("run_sensitive_fallback.py", old_mobile)


if __name__ == "__main__":
    unittest.main()
