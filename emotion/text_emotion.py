from transformers import pipeline

# NLP-based text emotion detection

class TextEmotionDetector:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

    def detect_emotion(self, text):
        if not text.strip():
            return "neutral"

        try:
            results = self.classifier(text)
            # handle both list and list of list structure
            if isinstance(results, list) and isinstance(results[0], list):
                result = results[0][0]
            else:
                result = results[0]
            return result["label"].lower()
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return "neutral"
