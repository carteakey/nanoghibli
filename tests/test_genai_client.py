import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from genai_client import resolve_genai_config, get_genai_client
import stylizer
import veo_animator


class TestGenaiClientConfig(unittest.TestCase):
    def setUp(self):
        self.env_backup = dict(os.environ)
        for key in [
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_GENAI_USE_ENTERPRISE",
            "VERTEXAI",
            "GOOGLE_CLOUD_PROJECT",
            "GCLOUD_PROJECT",
            "PROJECT_ID",
            "GOOGLE_CLOUD_LOCATION",
            "CLOUD_ML_REGION",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "VERTEX_API_KEY",
        ]:
            os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env_backup)

    def test_default_without_env_or_config(self):
        is_vertex, project, location, creds_path, api_key = resolve_genai_config()
        self.assertFalse(is_vertex)
        self.assertIsNone(project)
        self.assertIsNone(location)
        self.assertIsNone(creds_path)
        self.assertIsNone(api_key)

    def test_explicit_vertexai_flag_and_args(self):
        is_vertex, project, location, creds_path, api_key = resolve_genai_config(
            use_vertexai=True,
            project="test-project-123",
            location="europe-west1",
            credentials_path="/path/to/key.json",
        )
        self.assertTrue(is_vertex)
        self.assertEqual(project, "test-project-123")
        self.assertEqual(location, "europe-west1")
        self.assertEqual(creds_path, "/path/to/key.json")

    def test_default_location_when_vertex_enabled(self):
        is_vertex, project, location, creds_path, api_key = resolve_genai_config(
            use_vertexai=True,
            project="test-project-123",
        )
        self.assertTrue(is_vertex)
        self.assertEqual(location, "us-central1")

    def test_config_dict_resolution(self):
        cfg = {
            "google_cloud": {
                "use_vertexai": True,
                "project": "cfg-project",
                "location": "us-west1",
                "credentials": "/cfg/key.json",
            }
        }
        is_vertex, project, location, creds_path, api_key = resolve_genai_config(config=cfg)
        self.assertTrue(is_vertex)
        self.assertEqual(project, "cfg-project")
        self.assertEqual(location, "us-west1")
        self.assertEqual(creds_path, "/cfg/key.json")

    def test_environment_variables_resolution(self):
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = "env-project-456"
        os.environ["GOOGLE_CLOUD_LOCATION"] = "asia-northeast1"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/env/creds.json"

        is_vertex, project, location, creds_path, api_key = resolve_genai_config()
        self.assertTrue(is_vertex)
        self.assertEqual(project, "env-project-456")
        self.assertEqual(location, "asia-northeast1")
        self.assertEqual(creds_path, "/env/creds.json")

    def test_enterprise_env_var_alias(self):
        os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "1"
        os.environ["PROJECT_ID"] = "enterprise-proj"
        is_vertex, project, location, _, _ = resolve_genai_config()
        self.assertTrue(is_vertex)
        self.assertEqual(project, "enterprise-proj")
        self.assertEqual(location, "us-central1")

    def test_cli_overrides_config_and_env(self):
        os.environ["GOOGLE_CLOUD_PROJECT"] = "env-project"
        cfg = {"google_cloud": {"project": "cfg-project"}}
        is_vertex, project, _, _, _ = resolve_genai_config(
            project="cli-project",
            config=cfg,
        )
        self.assertEqual(project, "cli-project")

    def test_inferred_vertex_when_project_set_and_no_ai_studio_key(self):
        os.environ["GOOGLE_CLOUD_PROJECT"] = "trial-project-789"
        is_vertex, project, location, _, _ = resolve_genai_config()
        self.assertTrue(is_vertex)
        self.assertEqual(project, "trial-project-789")


class TestGetGenaiClient(unittest.TestCase):
    def setUp(self):
        self.env_backup = dict(os.environ)
        for key in [
            "GOOGLE_GENAI_USE_VERTEXAI",
            "GOOGLE_GENAI_USE_ENTERPRISE",
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ]:
            os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env_backup)

    def test_missing_auth_raises_helpful_error(self):
        with self.assertRaises(ValueError) as ctx:
            get_genai_client()
        self.assertIn("GEMINI_API_KEY", str(ctx.exception))
        self.assertIn("Google Cloud ($300 free trial / Vertex AI)", str(ctx.exception))

    def test_ai_studio_client_initialization(self):
        client = get_genai_client(api_key="fake-ai-studio-key")
        self.assertFalse(client._api_client.vertexai)

    def test_vertex_ai_client_initialization(self):
        client = get_genai_client(
            use_vertexai=True,
            project="my-gcp-trial-project",
            location="us-central1",
        )
        self.assertTrue(client._api_client.vertexai)
        self.assertEqual(client._api_client.project, "my-gcp-trial-project")
        self.assertEqual(client._api_client.location, "us-central1")

    def test_vertex_ai_missing_credentials_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            get_genai_client(
                use_vertexai=True,
                project="my-project",
                credentials_path="/nonexistent/path/to/sa.json",
            )


class TestClientPassingInModules(unittest.TestCase):
    def test_stylize_frames_accepts_client(self):
        mock_client = MagicMock()
        # Verify function accepts client keyword argument without error
        import inspect
        sig = inspect.signature(stylizer.stylize_frames)
        self.assertIn("client", sig.parameters)

    def test_veo_animator_accepts_client(self):
        import inspect
        sig = inspect.signature(veo_animator.generate_scene_video)
        self.assertIn("client", sig.parameters)


if __name__ == "__main__":
    unittest.main()
