import json
import os
import tempfile
import unittest
from pathlib import Path

from helpers.config_file import load_config_file


class ConfigFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_config(self, payload: object, name: str = "config.json") -> str:
        config_path = self.root / name
        if isinstance(payload, str):
            config_path.write_text(payload, encoding="utf-8")
        else:
            config_path.write_text(json.dumps(payload), encoding="utf-8")
        return str(config_path)

    def test_returns_declared_values(self) -> None:
        config_path = self._write_config(
            {
                "src": "/media/source",
                "dest": "/media/library",
                "model": "/models/category.onnx",
                "event_threshold": 20,
                "interactive": False,
                "family_group": "Household",
            }
        )

        values = load_config_file(config_path)

        self.assertEqual(values["src"], "/media/source")
        self.assertEqual(values["dest"], "/media/library")
        self.assertEqual(values["model"], "/models/category.onnx")
        self.assertEqual(values["event_threshold"], 20)
        self.assertFalse(values["interactive"])
        self.assertEqual(values["family_group"], "Household")

    def test_relative_paths_resolve_against_config_directory(self) -> None:
        config_path = self._write_config({"src": "photos", "dest": "../library"})

        values = load_config_file(config_path)

        self.assertEqual(values["src"], os.path.join(str(self.root), "photos"))
        self.assertEqual(values["dest"], os.path.normpath(str(self.root / ".." / "library")))

    def test_home_relative_paths_are_expanded(self) -> None:
        config_path = self._write_config({"references": "~/faces"})

        values = load_config_file(config_path)

        self.assertEqual(values["references"], os.path.join(str(Path.home()), "faces"))

    def test_null_values_are_treated_as_unset(self) -> None:
        config_path = self._write_config({"src": "/media/source", "references": None})

        values = load_config_file(config_path)

        self.assertIn("src", values)
        self.assertNotIn("references", values)

    def test_missing_file_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not exist"):
            load_config_file(str(self.root / "absent.json"))

    def test_invalid_json_is_rejected(self) -> None:
        config_path = self._write_config("{not json}")

        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            load_config_file(config_path)

    def test_non_object_json_is_rejected(self) -> None:
        config_path = self._write_config(["/media/source"])

        with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
            load_config_file(config_path)

    def test_unknown_key_is_rejected(self) -> None:
        config_path = self._write_config({"src": "/media/source", "destination": "/library"})

        with self.assertRaisesRegex(ValueError, "Unknown config key\\(s\\).*destination"):
            load_config_file(config_path)

    def test_non_integer_threshold_is_rejected(self) -> None:
        config_path = self._write_config({"event_threshold": "20"})

        with self.assertRaisesRegex(ValueError, "'event_threshold' must be an integer"):
            load_config_file(config_path)

    def test_boolean_threshold_is_rejected(self) -> None:
        config_path = self._write_config({"event_threshold": True})

        with self.assertRaisesRegex(ValueError, "'event_threshold' must be an integer"):
            load_config_file(config_path)

    def test_non_boolean_flag_is_rejected(self) -> None:
        config_path = self._write_config({"verbose": "yes"})

        with self.assertRaisesRegex(ValueError, "'verbose' must be true or false"):
            load_config_file(config_path)

    def test_non_string_path_is_rejected(self) -> None:
        config_path = self._write_config({"src": 5})

        with self.assertRaisesRegex(ValueError, "'src' must be a string"):
            load_config_file(config_path)


if __name__ == "__main__":
    unittest.main()
