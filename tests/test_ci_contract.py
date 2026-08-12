"""Regression checks for the clean-checkout CI contract."""

import os
import socket
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY")


class TestCIContract(unittest.TestCase):
    def test_provider_credentials_are_unset(self):
        present = [name for name in API_KEY_NAMES if os.getenv(name)]
        self.assertEqual(
            present,
            [],
            "account-free tests require provider credentials to be unset",
        )

    def test_network_guard_rejects_provider_connections(self):
        with self.assertRaisesRegex(AssertionError, "account-free"):
            socket.create_connection(("example.invalid", 443), timeout=0.01)

    def test_runtime_artifacts_are_not_tracked(self):
        result = subprocess.run(
            ["git", "ls-files", "--", "data", "*.mp4", "*.mov", "*.mkv"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout,
            "",
            "user media and generated runtime artifacts must remain ignored",
        )


if __name__ == "__main__":
    unittest.main()
