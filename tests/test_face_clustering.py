import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from pathlib import Path
from aisorter.models.face_identifier import FaceIdentifier


class FaceClusteringTest(unittest.TestCase):
    @patch("aisorter.models.face_identifier.ort.InferenceSession")
    @patch("aisorter.models.face_identifier._download_if_missing")
    def setUp(self, mock_download, mock_session) -> None:
        self.identifier = FaceIdentifier(model_path="dummy.onnx")
        self.identifier._references = {}
        self.identifier._reference_dir = Path("/dummy/references")
        # Mock the embedding method to return predefined vectors
        self.identifier._embed = MagicMock()

    @patch("aisorter.models.face_identifier.cv2.imwrite")
    @patch("aisorter.models.face_identifier.Path.mkdir")
    @patch("aisorter.models.face_identifier.Path.is_dir", return_value=False)
    def test_identify_no_match_saves_to_unknown(self, mock_is_dir, mock_mkdir, mock_imwrite) -> None:
        # Arrange
        self.identifier._references = {"Alice": np.array([1, 0])}
        self.identifier._embed.return_value = np.array([0, 1])  # Cosine distance = 1.0 (no match)
        face_crop = np.zeros((10, 10, 3), dtype=np.uint8)

        # Act
        result = self.identifier.identify(face_crop, source_path="img1.jpg", face_index=0)

        # Assert
        self.assertIsNone(result)
        mock_mkdir.assert_called_with(parents=True, exist_ok=True)
        # Verify it tries to write to the unknown directory
        expected_path = Path("/dummy/references/unknown/img1_face_0.jpg")
        mock_imwrite.assert_called_once()
        written_path = mock_imwrite.call_args[0][0]
        self.assertEqual(str(Path(written_path)), str(expected_path))

    @patch("aisorter.models.face_identifier.cv2.imread")
    @patch("aisorter.models.face_identifier.cv2.imwrite")
    @patch("aisorter.models.face_identifier.Path.mkdir")
    @patch("aisorter.models.face_identifier.Path.is_dir", return_value=True)
    @patch("aisorter.models.face_identifier.Path.exists", return_value=False)
    @patch("aisorter.models.face_identifier.Path.iterdir")
    @patch("aisorter.models.face_identifier.Path.rename")
    def test_identify_matches_unknown_creates_unknown_x(
        self, mock_rename, mock_iterdir, mock_exists, mock_is_dir, mock_mkdir, mock_imwrite, mock_imread
    ) -> None:
        # Arrange
        self.identifier._references = {"Alice": np.array([1, 0])}

        # Query face embedding
        query_emb = np.array([0, 1])
        # Matched face embedding (similar to query)
        matched_emb = np.array([0.01, 0.99])

        self.identifier._embed.side_effect = [query_emb, matched_emb]

        # Mock iterdir to yield one file in references/unknown/
        matched_file = Path("/dummy/references/unknown/old_img_face_1.jpg")
        mock_iterdir.return_value = [matched_file]

        # Mock reading the file from unknown/
        mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        face_crop = np.zeros((10, 10, 3), dtype=np.uint8)

        # Act
        result = self.identifier.identify(face_crop, source_path="new_img.jpg", face_index=2)

        # Assert
        self.assertEqual(result, "unknown_1")

        # Verify it creates the new directory for unknown_1
        new_dir = Path("/dummy/references/unknown_1")
        mock_mkdir.assert_any_call(parents=True, exist_ok=True)

        # Verify it renames/moves the old file to unknown_1/
        expected_rename_target = new_dir / "old_img_face_1.jpg"
        mock_rename.assert_called_once_with(expected_rename_target)

        # Verify it writes the new face to unknown_1/
        expected_imwrite_target = new_dir / "new_img_face_2.jpg"
        mock_imwrite.assert_called_once_with(str(expected_imwrite_target), face_crop)

        # Verify in-memory references dict was updated with the new identity
        self.assertIn("unknown_1", self.identifier._references)


if __name__ == "__main__":
    unittest.main()
