import json
from datetime import datetime
from collections import Counter
import csv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

#Vector stuff initiations
VECTOR_STORE_FILE = "vector_store.json"
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight embedding model

#Handles storing and retrieving conversation history

#initiating some files
SESSION_CONTEXT_FILE = "session_context.json"
MOOD_LOG_FILE = "mood_log.csv"





#Appends a new message with emotion to the session log.
def log_message(role, message, emotion):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "message": message,
        "emotion": emotion
    }
    
    try:
        with open(SESSION_CONTEXT_FILE, "r") as f:
            session = json.load(f)
    except FileNotFoundError:
        session = []
    
    session.append(entry)
    
    with open(SESSION_CONTEXT_FILE, "w") as f:
        json.dump(session, f, indent=2)

#Calculates the most frequent emotion in the session.
def get_overall_emotion():
    try:
        with open(SESSION_CONTEXT_FILE, "r") as f:
            session = json.load(f)
    except FileNotFoundError:
        return None
    
    emotions = [msg["emotion"] for msg in session if msg["role"] == "user"]
    
    if not emotions:
        return "neutral"
    
    emotion_counts = Counter(emotions)
    overall = emotion_counts.most_common(1)[0][0]
    
    return overall




def log_conversation_mood(overall_emotion):
    """Logs the overall mood of a conversation to a CSV file."""
    with open(MOOD_LOG_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), overall_emotion])


#Encodes and stores the message embedding.
def add_to_vector_store(message, role):
    embedding = model.encode(message).tolist()
    entry = {
        "role": role,
        "message": message,
        "embedding": embedding,
        "timestamp": datetime.now().isoformat()
    }

    try:
        with open(VECTOR_STORE_FILE, "r") as f:
            store = json.load(f)
    except FileNotFoundError:
        store = []

    store.append(entry)

    with open(VECTOR_STORE_FILE, "w") as f:
        json.dump(store, f, indent=2)

#Returns a formatted string of previous messages for use as conversation context
def get_formatted_context():
    try:
        with open("session_context.json", "r") as f:
            messages = json.load(f)
    except FileNotFoundError:
        return ""

    # Only include the last N messages (optional, e.g., 6)
    last_messages = messages[-4:]  

    context_lines = []
    for msg in last_messages:
        if msg["role"] == "user":
            context_lines.append(f"User: {msg['message']}")
        else:
            context_lines.append(f"Emil: {msg['message']}")
    
    return "\n".join(context_lines)

def get_relevant_context(user_query, top_k=3, vector_file= VECTOR_STORE_FILE):
    try:
        with open(vector_file, "r") as f:
            stored_data = json.load(f)
    except FileNotFoundError:
        return ""

    if not stored_data:
        return ""

    # Compute embedding for current user input
    query_embedding = model.encode([user_query])[0]

    # Compare similarity
    similarities = []
    for item in stored_data:
        stored_vector = np.array(item["embedding"])
        sim = cosine_similarity([query_embedding], [stored_vector])[0][0]
        similarities.append((sim, item["role"], item["message"]))

    # Sort and take top-k
    top_similar = sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]

    # Format nicely for LLM context
    context_lines = [f"{role.capitalize()}: {text}" for _, role, text in top_similar]
    return "\n".join(context_lines)

def get_past_emotion():
    emotions = pd.read_csv(MOOD_LOG_FILE)
    return emotions.iloc[-1,1]

def reset_session():
    open(SESSION_CONTEXT_FILE, 'w').write('[]')