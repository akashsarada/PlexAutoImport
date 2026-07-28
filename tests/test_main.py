import io
import io
import os
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
            family_group="Family",
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
        pipeline_class.return_value.process_image.return_value = {"is_family_photo": False}
        tag_video.return_value = {
            "is_family_photo": False,
            "frames_processed": 4,
        }
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
        image_writer = pipeline_class.return_value.process_image.call_args.kwargs[
            "keyword_writer"
        ]
        video_writer = tag_video.call_args.kwargs["keyword_writer"]
        self.assertIs(image_writer, video_writer)
        group_events.assert_called_once()
        self.assertEqual(group_events.call_args.args[1], 3)
        self.assertIs(group_events.call_args.args[2], move_file)
        stats = report_stats.call_args.args[0]
        self.assertEqual(stats.images_sorted, 1)
        self.assertEqual(stats.videos_sorted, 1)
        self.assertEqual(stats.total_entities, 2)
        self.assertEqual(stats.frames_processed, 5)
        self.assertEqual(stats.event_files_grouped, 2)
        self.assertEqual(stats.elapsed_seconds, 2.0)
        self.assertEqual(stats.entities_per_second, 1.0)
        self.assertEqual(stats.images_per_second, 2.5)
        self.assertEqual(stats.family_photos_sorted, 0)

    @patch("main.report_stats")
    @patch("main.group_events", return_value=0)
    @patch("main.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("main.move_file", return_value=True)
    @patch("main.progress", side_effect=lambda items, **_: items)
    @patch("main.AISorterPipeline")
    def test_family_photo_routed_to_family_dest(
        self,
        pipeline_class,
        _progress,
        move_file,
        _perf_counter,
        _group_events,
        _report_stats,
    ) -> None:
        (self.source / "photo.jpg").touch()
        pipeline_class.return_value.process_image.return_value = {"is_family_photo": True}
        family_dest = str(self.root / "Family Photos")
        config = main.RuntimeConfig(
            src=str(self.source),
            dest=str(self.root / "destination"),
            category_model=str(self.category_model),
            face_detector_model=None,
            references=None,
            face_identifier_model=None,
            family_dest=family_dest,
        )

        main.run_import(config, verbose=False)

        dest_arg = move_file.call_args.args[1]
        self.assertTrue(dest_arg.startswith(family_dest))

    @patch("main.report_stats")
    @patch("main.group_events", return_value=0)
    @patch("main.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("main.move_file", return_value=True)
    @patch("main.progress", side_effect=lambda items, **_: items)
    @patch("main.AISorterPipeline")
    def test_family_photo_routed_to_default_family_dest(
        self,
        pipeline_class,
        _progress,
        move_file,
        _perf_counter,
        _group_events,
        _report_stats,
    ) -> None:
        (self.source / "photo.jpg").touch()
        pipeline_class.return_value.process_image.return_value = {"is_family_photo": True}
        dest = str(self.root / "destination")
        config = main.RuntimeConfig(
            src=str(self.source),
            dest=dest,
            category_model=str(self.category_model),
            face_detector_model=None,
            references=None,
            face_identifier_model=None,
        )

        main.run_import(config, verbose=False)

        dest_arg = move_file.call_args.args[1]
        expected_prefix = os.path.join(dest, "Family Photos")
        self.assertTrue(dest_arg.startswith(expected_prefix))

    @patch("main.report_stats")
    @patch("main.group_events", return_value=0)
    @patch("main.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("main.move_file", return_value=True)
    @patch("main.progress", side_effect=lambda items, **_: items)
    @patch("main.AISorterPipeline")
    def test_family_photos_sorted_counted_in_stats(
        self,
        pipeline_class,
        _progress,
        _move_file,
        _perf_counter,
        _group_events,
        report_stats,
    ) -> None:
        (self.source / "fam1.jpg").touch()
        (self.source / "fam2.jpg").touch()
        (self.source / "regular.jpg").touch()
        pipeline_class.return_value.process_image.side_effect = [
            {"is_family_photo": True},
            {"is_family_photo": True},
            {"is_family_photo": False},
        ]
        config = main.RuntimeConfig(
            src=str(self.source),
            dest=str(self.root / "destination"),
            category_model=str(self.category_model),
            face_detector_model=None,
            references=None,
            face_identifier_model=None,
        )

        main.run_import(config, verbose=False)

        stats = report_stats.call_args.args[0]
        self.assertEqual(stats.family_photos_sorted, 2)
        self.assertEqual(stats.images_sorted, 3)

    def test_report_stats_prints_summary_in_quiet_mode(self) -> None:
        output = io.StringIO()
        stats = main.ImportStats(
            images_sorted=4,
            videos_sorted=1,
            elapsed_seconds=2.0,
            frames_processed=10,
            event_files_grouped=3,
        )

        with self.assertLogs("main", level="INFO") as logs, redirect_stdout(output):
            main.report_stats(stats, verbose=False)

        self.assertIn("Images sorted: 4", output.getvalue())
        self.assertIn("Videos sorted: 1", output.getvalue())
        self.assertIn("Frames processed: 10", output.getvalue())
        self.assertIn("Event files grouped: 3", output.getvalue())
        self.assertIn("Entities per second: 2.50", output.getvalue())
        self.assertIn("Images per second: 5.00", output.getvalue())
        self.assertIn("Elapsed time: 2.00 seconds", "\n".join(logs.output))

    def test_report_stats_relies_on_logs_in_verbose_mode(self) -> None:
        output = io.StringIO()
        stats = main.ImportStats(images_sorted=0, videos_sorted=0, elapsed_seconds=0.0)

        with self.assertLogs("main", level="INFO") as logs, redirect_stdout(output):
            main.report_stats(stats, verbose=True)

        self.assertEqual(output.getvalue(), "")
        log_output = "\n".join(logs.output)
        self.assertIn("Entities per second: 0.00", log_output)
        self.assertIn("Images per second: 0.00", log_output)

    @patch("main.write_keywords")
    @patch("main.extract_frames_from_video", return_value=["frame_0.jpg", "frame_1.jpg"])
    def test_tag_video_passes_write_metadata_false_to_frames(
        self, _extract_frames, write_keywords
    ) -> None:
        pipeline = Mock()
        pipeline.process_image.return_value = {
            "categories": ["nature"],
            "identities": [],
            "is_family_photo": False,
        }

        result = main._tag_video(pipeline, "/video/clip.mp4", "clip.mp4")

        calls = pipeline.process_image.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["frames_processed"], 2)
        for call in calls:
            self.assertEqual(call.kwargs.get("write_metadata"), False)

    @patch("main.write_keywords")
    @patch("main.extract_frames_from_video", return_value=["frame_0.jpg", "frame_1.jpg"])
    def test_tag_video_excludes_failed_frames_from_processed_count(
        self,
        _extract_frames,
        _write_keywords,
    ) -> None:
        pipeline = Mock()
        pipeline.process_image.side_effect = [
            {
                "categories": ["nature"],
                "identities": [],
                "is_family_photo": False,
            },
            RuntimeError("inference failed"),
        ]

        with self.assertLogs("main", level="ERROR"):
            result = main._tag_video(pipeline, "/video/clip.mp4", "clip.mp4")

        self.assertEqual(result["frames_processed"], 1)

    @patch("main.write_keywords")
    @patch("main.extract_frames_from_video", return_value=["frame_0.jpg"])
    def test_tag_video_writes_metadata_once_to_source_video(
        self, _extract_frames, write_keywords
    ) -> None:
        pipeline = Mock()
        pipeline.process_image.return_value = {
            "categories": ["nature"],
            "identities": ["Alice"],
            "is_family_photo": False,
        }

        main._tag_video(pipeline, "/video/clip.mp4", "clip.mp4")

        write_keywords.assert_called_once()
        path_arg = write_keywords.call_args.args[0]
        self.assertEqual(path_arg, "/video/clip.mp4")


    @patch("main.write_keywords")
    @patch("main.extract_frames_from_video", return_value=["frame_0.jpg"])
    def test_tag_video_uses_supplied_keyword_writer(
        self,
        _extract_frames,
        write_keywords,
    ) -> None:
        pipeline = Mock()
        pipeline.process_image.return_value = {
            "categories": ["nature"],
            "identities": ["Alice"],
            "is_family_photo": False,
        }
        keyword_writer = Mock()

        main._tag_video(
            pipeline,
            "/video/clip.mp4",
            "clip.mp4",
            keyword_writer=keyword_writer,
        )

        keyword_writer.assert_called_once_with(
            "/video/clip.mp4",
            ["Alice", "nature"],
        )
        write_keywords.assert_not_called()


    @patch("main.report_stats")
    @patch("main.group_events", return_value=0)
    @patch("main.time.perf_counter", side_effect=[0.0, 1.0])
    @patch("main.move_file", return_value=True)
    @patch("main.progress", side_effect=lambda items, **_: items)
    @patch("main.ExifToolKeywordWriter")
    @patch("main.AISorterPipeline")
    def test_keyword_write_failure_returns_nonzero_exit(
        self,
        pipeline_class,
        writer_class,
        _progress,
        move_file,
        _perf_counter,
        _group_events,
        _report_stats,
    ) -> None:
        (self.source / "photo.jpg").touch()
        keyword_writer = writer_class.return_value.__enter__.return_value
        keyword_writer.side_effect = RuntimeError("metadata write failed")

        def process_image(
            image_path: str,
            *,
            keyword_writer: main.WriteKeywords,
        ) -> None:
            keyword_writer(image_path, ["nature"])

        pipeline_class.return_value.process_image.side_effect = process_image
        config = main.RuntimeConfig(
            src=str(self.source),
            dest=str(self.root / "destination"),
            category_model=str(self.category_model),
            face_detector_model=None,
            references=None,
            face_identifier_model=None,
        )

        result = main.run_import(config, verbose=False)

        self.assertEqual(result, 1)
        move_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
