import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_checks import blend_detection_scores


def test_blend_detection_scores_keeps_strong_face_crop_signal():
    score = blend_detection_scores(full_frame=0.18, face_crop=0.91, has_face=True)
    assert 0.45 < score < 0.75


def test_blend_detection_scores_falls_back_to_full_frame_when_no_face_found():
    score = blend_detection_scores(full_frame=0.82, face_crop=0.10, has_face=False)
    assert score == pytest.approx(0.82, abs=1e-6)
