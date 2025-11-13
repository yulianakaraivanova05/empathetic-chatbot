#combines detection into a final emotion

def fuse_emotions(facial_emotion, text_emotion):
    #Simple rule-based fusion of facial and text emotions.
    
    # If they agree, return either
    if facial_emotion == text_emotion:
        return facial_emotion

    # Example priority rules
    priority = ["anger", "fear", "sadness", "disgust", "surprise", "happy", "neutral"]

    if facial_emotion in priority and text_emotion in priority:
        # Return the one with higher emotional priority (lower index)
        return facial_emotion if priority.index(facial_emotion) < priority.index(text_emotion) else text_emotion

    return text_emotion  # fallback