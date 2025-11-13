# tests/test_emotions.py
import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from emotion.text_emotion import TextEmotionDetector
from emotion.facial_emotion import VideoRec

# -----------------------------
# TEXT EMOTION TESTS
# -----------------------------
@pytest.fixture(scope="module")
def text_detector():
    return TextEmotionDetector()

@pytest.mark.parametrize("text,expected", [
    ("I am so happy today!", "joy"),
    ("I want to cry.", "sadness"),
    ("I'm really angry about this situation!", "anger"),
    ("That was terrifying, I was so scared.", "fear"),
    ("Wow, I did not expect that!", "surprise"),
    ("I feel disgusted just thinking about it.", "disgust"),
    ("", "neutral"),
])
def test_text_emotions(text_detector, text, expected):
    prediction = text_detector.detect_emotion(text)
    assert prediction == expected, f"Expected {expected}, got {prediction}"


# -----------------------------
# FACIAL EMOTION TESTS
# -----------------------------
@pytest.fixture(scope="module")
def video_rec():
    return VideoRec()



@pytest.mark.parametrize("video_path,expected", [
    ("tests/test_videos/happy.mp4", "happy"),
    ("tests/test_videos/sad.mp4", "sad"),
    ("tests/test_videos/angry.mp4", "angry"),
    ("tests/test_videos/fear.mp4", "fear"),
    ("tests/test_videos/surprise.mp4", "surprise"),
    ("tests/test_videos/disgust.mp4", "disgust"),
    ("tests/test_videos/neutral.mp4", "neutral"),
])
def test_facial_emotions(video_rec, video_path, expected):
    """Tests that facial emotion recognition from videos works as expected."""
    assert os.path.exists(video_path), f"Missing test video: {video_path}"

    results = video_rec.extract_emotion(video_path)

    # Validate output structure
    assert isinstance(results, dict), f"Expected dict, got {type(results)}"
    assert "deepface" in results and "custom" in results, (
        f"Expected keys 'deepface' and 'custom' in results, got {list(results.keys())}"
    )

    # At least one classifier should correctly identify the expected emotion
    assert (
        results["deepface"] == expected or results["custom"] == expected
    ), f"Expected {expected}, got DeepFace={results['deepface']} Custom={results['custom']}"

