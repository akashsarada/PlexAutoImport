import io
import logging
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from helpers.runtime_output import configure_logging, progress


class RuntimeOutputTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.temp_dir.name) / "import.log"

    def tearDown(self) -> None:
        root_logger = logging.getLogger()
        for handler in list(root_logger.handlers):
            if getattr(handler, "name", None) == "plex_auto_import":
                root_logger.removeHandler(handler)
                handler.close()
        self.temp_dir.cleanup()

    def test_quiet_mode_logs_to_file_without_terminal_output(self) -> None:
        terminal = io.StringIO()
        with redirect_stderr(terminal):
            configure_logging(False, str(self.log_file))
            logging.getLogger("test.quiet").info("quiet message")
            self._flush_handlers()

        self.assertIn("quiet message", self.log_file.read_text(encoding="utf-8"))
        self.assertEqual(terminal.getvalue(), "")

    def test_verbose_mode_logs_to_file_and_terminal(self) -> None:
        terminal = io.StringIO()
        with redirect_stderr(terminal):
            configure_logging(True, str(self.log_file))
            logging.getLogger("test.verbose").info("verbose message")
            self._flush_handlers()

        self.assertIn("verbose message", self.log_file.read_text(encoding="utf-8"))
        self.assertIn("INFO verbose message", terminal.getvalue())

    @patch("helpers.runtime_output.tqdm")
    def test_progress_bar_is_enabled_only_in_quiet_mode(self, tqdm_mock) -> None:
        items = ["one", "two"]
        tqdm_mock.return_value = items

        self.assertEqual(list(progress(items, verbose=False, description="Test", unit="item")), items)
        self.assertFalse(tqdm_mock.call_args.kwargs["disable"])

        self.assertEqual(list(progress(items, verbose=True, description="Test", unit="item")), items)
        self.assertTrue(tqdm_mock.call_args.kwargs["disable"])

    @staticmethod
    def _flush_handlers() -> None:
        for handler in logging.getLogger().handlers:
            handler.flush()


if __name__ == "__main__":
    unittest.main()
