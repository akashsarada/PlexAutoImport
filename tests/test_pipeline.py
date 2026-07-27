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


if __name__ == "__main__":
    unittest.main()
