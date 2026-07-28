import unittest
from unittest.mock import Mock, patch

import numpy as np

from aisorter.pipeline import AISorterPipeline


class PipelineLoggingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = object.__new__(AISorterPipeline)
        self.pipeline._category_input_name = "input"
        self.pipeline._category_session = Mock()
        self.pipeline._face_detector = Mock()
        self.pipeline._face_identifier = Mock()
        self.pipeline._family_identities = set()
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        self.pipeline._load_image = Mock(return_value=image)

    @patch("aisorter.pipeline.write_keywords")
    def test_logs_completion_after_all_executed_stages(self, write_keywords) -> None:
        self.pipeline._category_session.run.return_value = [
            np.array([[10.0, 10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = [
            {"bbox": (0.0, 0.0, 4.0, 4.0), "confidence": 0.9}
        ]
        self.pipeline._face_identifier.identify.return_value = "Alice"

        with self.assertLogs("aisorter.pipeline", level="INFO") as logs:
            result = self.pipeline.process_image("photo.jpg")

        output = "\n".join(logs.output)
        self.assertIn("Stage 1/4 face detection complete", output)
        self.assertIn("Stage 2/4 face identification complete", output)
        self.assertIn("Stage 3/4 category classification complete", output)
        self.assertIn("Stage 4/4 keyword write complete", output)
        self.assertEqual(result["identities"], ["Alice"])
        write_keywords.assert_called_once_with("photo.jpg", ["Alice", "cars", "people"])

    @patch("aisorter.pipeline.write_keywords")
    def test_logs_skipped_conditional_stages(self, _write_keywords) -> None:
        self.pipeline._category_session.run.return_value = [
            np.array([[-10.0, -10.0, 10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = []
        self.pipeline._face_identifier = None

        with self.assertLogs("aisorter.pipeline", level="INFO") as logs:
            self.pipeline.process_image("scenery.jpg")

        output = "\n".join(logs.output)
        self.assertIn("Stage 1/4 face detection complete", output)
        self.assertIn("Stage 2/4 face identification skipped", output)
        self.assertIn("Stage 3/4 category classification complete", output)
        self.assertIn("Stage 4/4 keyword write complete", output)

    @patch("aisorter.pipeline.write_keywords")
    def test_family_keyword_added_when_two_family_members_identified(self, write_keywords) -> None:
        self.pipeline._family_identities = {"Alice", "Bob"}
        self.pipeline._category_session.run.return_value = [
            np.array([[-10.0, 10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = [
            {"bbox": (0.0, 0.0, 4.0, 4.0), "confidence": 0.9},
            {"bbox": (5.0, 0.0, 8.0, 4.0), "confidence": 0.9},
        ]
        self.pipeline._face_identifier.identify.side_effect = ["Alice", "Bob"]

        result = self.pipeline.process_image("family.jpg")

        self.assertTrue(result["is_family_photo"])
        labels = write_keywords.call_args.args[1]
        self.assertIn("Family", labels)

    @patch("aisorter.pipeline.write_keywords")
    def test_family_keyword_not_added_when_only_one_family_member(self, write_keywords) -> None:
        self.pipeline._family_identities = {"Alice", "Bob"}
        self.pipeline._category_session.run.return_value = [
            np.array([[-10.0, 10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = [
            {"bbox": (0.0, 0.0, 4.0, 4.0), "confidence": 0.9},
        ]
        self.pipeline._face_identifier.identify.return_value = "Alice"

        result = self.pipeline.process_image("solo.jpg")

        self.assertFalse(result["is_family_photo"])
        labels = write_keywords.call_args.args[1]
        self.assertNotIn("Family", labels)

    @patch("aisorter.pipeline.write_keywords")
    def test_family_keyword_not_duplicated_when_already_present(self, write_keywords) -> None:
        self.pipeline._family_identities = {"Family", "Bob"}
        self.pipeline._category_session.run.return_value = [
            np.array([[-10.0, 10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = [
            {"bbox": (0.0, 0.0, 4.0, 4.0), "confidence": 0.9},
            {"bbox": (5.0, 0.0, 8.0, 4.0), "confidence": 0.9},
        ]
        self.pipeline._face_identifier.identify.side_effect = ["Family", "Bob"]

        result = self.pipeline.process_image("photo.jpg")

        self.assertTrue(result["is_family_photo"])
        labels = write_keywords.call_args.args[1]
        self.assertEqual(labels.count("Family"), 1)

    @patch("aisorter.pipeline.write_keywords")
    def test_write_metadata_true_calls_write_keywords(self, write_keywords) -> None:
        self.pipeline._category_session.run.return_value = [
            np.array([[10.0, -10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = []
        self.pipeline._face_identifier = None

        result = self.pipeline.process_image("photo.jpg", write_metadata=True)

        write_keywords.assert_called_once_with("photo.jpg", result["labels"])

    @patch("aisorter.pipeline.write_keywords")
    def test_write_metadata_false_skips_write_keywords(self, write_keywords) -> None:
        self.pipeline._category_session.run.return_value = [
            np.array([[10.0, -10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = []
        self.pipeline._face_identifier = None

        self.pipeline.process_image("photo.jpg", write_metadata=False)

        write_keywords.assert_not_called()

    @patch("aisorter.pipeline.write_keywords")
    def test_write_metadata_false_returns_identical_analysis(self, _write_keywords) -> None:
        self.pipeline._category_session.run.return_value = [
            np.array([[10.0, 10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = [
            {"bbox": (0.0, 0.0, 4.0, 4.0), "confidence": 0.9}
        ]
        self.pipeline._face_identifier.identify.return_value = "Alice"

        result_write = self.pipeline.process_image("photo.jpg", write_metadata=True)
        result_skip = self.pipeline.process_image("photo.jpg", write_metadata=False)

        for key in ("faces", "identities", "categories", "labels", "is_family_photo"):
            self.assertEqual(result_write[key], result_skip[key])

    @patch("aisorter.pipeline.write_keywords")
    def test_write_metadata_false_logs_stage_4_skipped(self, _write_keywords) -> None:
        self.pipeline._category_session.run.return_value = [
            np.array([[-10.0, -10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = []
        self.pipeline._face_identifier = None

        with self.assertLogs("aisorter.pipeline", level="INFO") as logs:
            self.pipeline.process_image("frame.jpg", write_metadata=False)

        output = "\n".join(logs.output)
        self.assertIn("Stage 4/4 keyword write skipped", output)
        self.assertIn("write_metadata=False", output)

    @patch("aisorter.pipeline.write_keywords")
    def test_stage_logs_include_elapsed_ms(self, _write_keywords) -> None:
        self.pipeline._category_session.run.return_value = [
            np.array([[-10.0, -10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = []
        self.pipeline._face_identifier = None

        with self.assertLogs("aisorter.pipeline", level="INFO") as logs:
            self.pipeline.process_image("photo.jpg")

        output = "\n".join(logs.output)
        self.assertIn("elapsed_ms=", output)
        stage_lines = [l for l in logs.output if "Stage" in l]
        self.assertEqual(len(stage_lines), 4)
        for line in stage_lines:
            self.assertIn("elapsed_ms=", line)


    @patch("aisorter.pipeline.write_keywords")
    def test_process_image_uses_supplied_keyword_writer(self, write_keywords) -> None:
        self.pipeline._category_session.run.return_value = [
            np.array([[10.0, -10.0, -10.0]], dtype=np.float32)
        ]
        self.pipeline._face_detector.detect.return_value = []
        self.pipeline._face_identifier = None
        keyword_writer = Mock()

        result = self.pipeline.process_image(
            "photo.jpg",
            keyword_writer=keyword_writer,
        )

        keyword_writer.assert_called_once_with("photo.jpg", result["labels"])
        write_keywords.assert_not_called()

    @patch("aisorter.pipeline.tqdm", side_effect=lambda items, **_: items)
    @patch(
        "aisorter.pipeline.os.walk",
        return_value=[("photos", [], ["one.jpg", "two.jpg"])],
    )
    @patch("aisorter.pipeline.ExifToolKeywordWriter")
    def test_process_directory_reuses_keyword_writer(
        self,
        writer_class,
        _walk,
        _tqdm,
    ) -> None:
        pipeline = object.__new__(AISorterPipeline)
        pipeline.process_image = Mock(return_value={})
        keyword_writer = Mock()
        writer_class.return_value.__enter__.return_value = keyword_writer

        pipeline.process_directory("photos")

        self.assertEqual(pipeline.process_image.call_count, 2)
        for process_call in pipeline.process_image.call_args_list:
            self.assertIs(process_call.kwargs["keyword_writer"], keyword_writer)
        writer_class.return_value.__exit__.assert_called_once_with(None, None, None)


if __name__ == "__main__":
    unittest.main()
