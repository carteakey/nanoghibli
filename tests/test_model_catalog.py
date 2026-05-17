import os
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from main import load_config
from model_catalog import (
    model_cache_slug,
    model_provider,
    normalize_image_grid_model,
    normalize_image_grid_models,
    normalize_stylizer_model,
    normalize_stylizer_models,
    rotation_signature,
)


class TestModelCatalog(unittest.TestCase):
    def test_aliases_normalize_to_model_ids(self):
        self.assertEqual(
            normalize_stylizer_model("flash"),
            "gemini-3.1-flash-image-preview",
        )
        self.assertEqual(
            normalize_stylizer_model("nano-banana"),
            "gemini-2.5-flash-image",
        )
        self.assertEqual(
            normalize_stylizer_model("nano-banana-pro-2k"),
            "gemini-3-pro-image-preview-2k",
        )

    def test_rotation_list_parses_commas_and_lists(self):
        self.assertEqual(
            normalize_stylizer_models(["flash, pro-2k", "pro"]),
            [
                "gemini-3.1-flash-image-preview",
                "gemini-3-pro-image-preview-2k",
                "gemini-3-pro-image-preview",
            ],
        )

    def test_openai_high_fidelity_alias_maps_to_api_model(self):
        self.assertEqual(
            normalize_stylizer_model("gpt-image-1.5-high-fidelity"),
            "gpt-image-1.5",
        )
        self.assertEqual(model_provider("gpt-image-1.5"), "openai")

    def test_manual_chatgpt_alias_maps_to_manual_provider(self):
        self.assertEqual(
            normalize_stylizer_model("chatgpt-manual"),
            "manual-chatgpt-images",
        )
        self.assertEqual(model_cache_slug("manual-chatgpt-images"), "chatgpt_manual")
        self.assertEqual(model_provider("manual-chatgpt-images"), "manual")

    def test_cache_slugs_are_distinct(self):
        self.assertEqual(model_cache_slug("gemini-3-pro-image-preview"), "pro")
        self.assertEqual(model_cache_slug("gemini-3-pro-image-preview-2k"), "pro_2k")
        self.assertEqual(model_cache_slug("imagen-4.0-fast-generate-001"), "imagen_4_fast")

    def test_image_grid_aliases_include_imagen_and_nano_banana(self):
        self.assertEqual(
            normalize_image_grid_model("nano-banana-2"),
            "gemini-3.1-flash-image-preview",
        )
        self.assertEqual(
            normalize_image_grid_model("nano-banana"),
            "gemini-2.5-flash-image",
        )
        self.assertEqual(
            normalize_image_grid_model("imagen-4-ultra"),
            "imagen-4.0-ultra-generate-001",
        )
        self.assertEqual(
            normalize_image_grid_models(["nano-banana-pro, imagen-4-fast"]),
            [
                "gemini-3-pro-image-preview",
                "imagen-4.0-fast-generate-001",
            ],
        )
        self.assertEqual(model_provider("imagen-4.0-generate-001"), "google")
        self.assertEqual(model_provider("gemini-2.5-flash-image"), "google")

    def test_rotation_signature_changes_for_mixed_models(self):
        single = rotation_signature(["gemini-3.1-flash-image-preview"])
        mixed = rotation_signature([
            "gemini-3.1-flash-image-preview",
            "gemini-3-pro-image-preview-2k",
        ])
        self.assertEqual(single, "flash")
        self.assertTrue(mixed.startswith("rot_"))

    def test_empty_config_loads_as_empty_dict(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            self.assertEqual(load_config(path), {})
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
