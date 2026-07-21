import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import main


class MainConfigurationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.category_model = self.root / "category.onnx"
        self.category_model.touch()
        self.face_detector_model = self.root / "detector.onnx"
        self.face_detector_model.touch()
        self.references = self.root / "references"
        self.references.mkdir()
        self.face_identifier_model = self.root / "identifier.onnx"
        self.face_identifier_model.touch()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_supplied_locations_skip_prompts(self) -> None:
        destination = self.root / "destination"
        args = main.parse_args([
            str(self.source),
            str(destination),
            str(self.category_model),
            "--face-detector-model",
            str(self.face_detector_model),
            "--references",
            str(self.references),
            "--face-identifier-model",
            str(self.face_identifier_model),
        ])
        prompt = Mock(side_effect=AssertionError("prompt should not be called"))

        config = main.resolve_config(args, prompt)

        self.assertEqual(config.src, str(self.source.resolve()))
        self.assertEqual(config.dest, str(destination.resolve()))
        self.assertEqual(config.category_model, str(self.category_model.resolve()))
        self.assertEqual(config.face_detector_model, str(self.face_detector_model.resolve()))
        self.assertEqual(config.references, str(self.references.resolve()))
        self.assertEqual(config.face_identifier_model, str(self.face_identifier_model.resolve()))
        prompt.assert_not_called()

    def test_missing_locations_are_prompted_by_default(self) -> None:
        destination = self.root / "destination"
        prompt = Mock(side_effect=[
            str(self.source),
            str(destination),
            str(self.category_model),
            "",
            "",
        ])

        config = main.resolve_config(main.parse_args([]), prompt)

        self.assertEqual(config.src, str(self.source.resolve()))
        self.assertEqual(config.dest, str(destination.resolve()))
        self.assertEqual(config.category_model, str(self.category_model.resolve()))
        self.assertIsNone(config.face_detector_model)
        self.assertIsNone(config.references)
        self.assertIsNone(config.face_identifier_model)
        self.assertEqual(prompt.call_count, 5)

    def test_face_identifier_is_prompted_when_references_are_enabled(self) -> None:
        destination = self.root / "destination"
        prompt = Mock(side_effect=[
            str(self.source),
            str(destination),
            str(self.category_model),
            str(self.face_detector_model),
            str(self.references),
            str(self.face_identifier_model),
        ])

        config = main.resolve_config(main.parse_args([]), prompt)

        self.assertEqual(config.face_identifier_model, str(self.face_identifier_model.resolve()))
        self.assertEqual(prompt.call_count, 6)

    def test_invalid_event_threshold_is_rejected(self) -> None:
        args = main.parse_args([
            str(self.source),
            str(self.root / "destination"),
            str(self.category_model),
            "--event-threshold",
            "0",
            "--no-interactive",
        ])

        with self.assertRaisesRegex(ValueError, "Event threshold"):
            main.resolve_config(args)

    def test_no_interactive_rejects_missing_required_locations(self) -> None:
        args = main.parse_args(["--no-interactive"])

        with self.assertRaisesRegex(ValueError, "Source folder"):
            main.resolve_config(args)

    @patch("main.report_stats")
    @patch("main.progress", side_effect=lambda items, **_: items)
    @patch("main.AISorterPipeline")
    def test_run_import_forwards_all_model_locations(
        self,
        pipeline_class,
        _progress,
        _report_stats,
    ) -> None:
        destination = self.root / "destination"
        config = main.RuntimeConfig(
            src=str(self.source),
            dest=str(destination),
            category_model=str(self.category_model),
            face_detector_model=str(self.face_detector_model),
            references=str(self.references),
            face_identifier_model=str(self.face_identifier_model),
        )

        result = main.run_import(config, verbose=False)

        self.assertEqual(result, 0)
        pipeline_class.assert_called_once_with(
            reference_dir=str(self.references),
            category_model_path=str(self.category_model),
            face_detector_model_path=str(self.face_detector_model),
            face_identifier_model_path=str(self.face_identifier_model),
        )


    @patch("main.report_stats")
    @patch("main.group_events", return_value=2)
    @patch("main.time.perf_counter", side_effect=[10.0, 12.0])
    @patch("main.move_file", return_value=True)
    @patch("main._tag_video")
    @patch("main.progress", side_effect=lambda items, **_: items)
    @patch("main.AISorterPipeline")
    def test_run_import_reports_sorted_media_stats(
        self,
        pipeline_class,
        _progress,
        tag_video,
        move_file,
        _perf_counter,
        group_events,
        report_stats,
    ) -> None:
        (self.source / "photo.jpg").touch()
        (self.source / "clip.mp4").touch()
        (self.source / "notes.txt").touch()
        config = main.RuntimeConfig(
            src=str(self.source),
            dest=str(self.root / "destination"),
            category_model=str(self.category_model),
            face_detector_model=None,
            references=None,
            face_identifier_model=None,
            event_threshold=3,
        )

        result = main.run_import(config, verbose=False)

        self.assertEqual(result, 0)
        pipeline_class.return_value.process_image.assert_called_once()
        tag_video.assert_called_once()
        group_events.assert_called_once()
        self.assertEqual(group_events.call_args.args[1], 3)
        self.assertIs(group_events.call_args.args[2], move_file)
        stats = report_stats.call_args.args[0]
        self.assertEqual(stats.images_sorted, 1)
        self.assertEqual(stats.videos_sorted, 1)
        self.assertEqual(stats.total_entities, 2)
        self.assertEqual(stats.event_files_grouped, 2)
        self.assertEqual(stats.elapsed_seconds, 2.0)
        self.assertEqual(stats.entities_per_second, 1.0)

    def test_report_stats_prints_summary_in_quiet_mode(self) -> None:
        output = io.StringIO()
        stats = main.ImportStats(
            images_sorted=4,
            videos_sorted=1,
            elapsed_seconds=2.0,
            event_files_grouped=3,
        )

        with self.assertLogs("main", level="INFO") as logs, redirect_stdout(output):
            main.report_stats(stats, verbose=False)

        self.assertIn("Images sorted: 4", output.getvalue())
        self.assertIn("Videos sorted: 1", output.getvalue())
        self.assertIn("Event files grouped: 3", output.getvalue())
        self.assertIn("Entities per second: 2.50", output.getvalue())
        self.assertIn("Elapsed time: 2.00 seconds", "\n".join(logs.output))

    def test_report_stats_relies_on_logs_in_verbose_mode(self) -> None:
        output = io.StringIO()
        stats = main.ImportStats(images_sorted=0, videos_sorted=0, elapsed_seconds=0.0)

        with self.assertLogs("main", level="INFO") as logs, redirect_stdout(output):
            main.report_stats(stats, verbose=True)

        self.assertEqual(output.getvalue(), "")
        self.assertIn("Entities per second: 0.00", "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
