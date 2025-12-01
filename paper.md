---
title: 'Emil: An Emotion-Aware, Empathy-Driven Chatbot with Multi-Modal Emotion Detection'
tags:
  - Python
  - chatbot
  - emotion recognition
  - NLP
  - LLaMA3
  - empathy
authors:
  - name: Yuliana Karaivanova
    orcid: 0009-0005-0575-4172
    affiliation: 1
  - name: Barry Hogan
    orcid:
    affiliation: 1
  - name: Daniel J. Finnegan
    orcid: 0000-0003-1169-2842
    affiliation: 1
  - name: Yipeng Qin
    orcid: 0000-0002-1551-9126
    affiliation: 1
affiliations:
    - index: 1
      name: School of Computer Science and Informatics, Cardiff University
date: 29 November 2025
bibliography: paper.bib
---

# Summary

Emil is an emotion-aware, empathy-driven chatbot that integrates multi-modal emotion detection with context-aware language generation. It processes **textual** and **visual** emotional cues in real time, combining facial expression recognition and natural language processing into a unified emotional state via a rule-based fusion strategy. This emotion, along with conversation history and relevant prior interactions, is injected into prompts for a locally hosted [LLaMA3](https://ai.meta.com/blog/meta-llama-3/) model (via [Ollama](https://ollama.com/)), enabling **privacy-preserving** (by deploying locally instead of using an online service API), **low-latency**, and **empathetic** responses.

Unlike traditional chatbots, Emil incorporates **user-personalized facial emotion classifiers** to adapt to individual expressiveness, as well as **Person-Centred Therapy (PCT) principles** in prompt engineering to enhance empathetic engagement.

# Statement of need

Conversational AI systems often lack **emotional intelligence** and the ability to adapt responses based on nuanced human affect. While emotion recognition has been explored in isolation, few open-source projects provide an **integrated, privacy-preserving pipeline** for multi-modal emotion detection fused with real-time empathetic language generation.

Emil addresses this gap by:
- Combining facial and textual emotion detection into a single, interpretable emotional state.
- Supporting personalization via per-user facial emotion training.
- Running locally to ensure **data privacy** and reduce dependency on cloud services.
- Embedding therapeutic communication strategies to improve user emotional experience.

This makes Emil valuable for research on **mental health support tools**, **empathy-driven interfaces**, and **affective computing**.

# Implementation

The Emil system is structured as an integrated pipeline with the following components:

1. **Input Capture** – Webcam video and user text are collected during typing events, ensuring naturalistic data gathering.
2. **Facial Emotion Recognition** – Real-time analysis using [DeepFace](https://github.com/serengil/deepface) for general emotion classification, combined with a custom-trained Support Vector Machine classifier for user-personalized accuracy, leveraging [MediaPipe FaceMesh](https://developers.google.com/mediapipe) for feature extraction.
3. **Text Emotion Analysis** – NLP-based emotion detection using a fine-tuned DistilRoBERTa model ([`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)) from HuggingFace Transformers.
4. **Emotion Fusion** – A rule-based strategy prioritizing affective salience combines facial and textual cues into a single emotional state.
5. **Response Generation** – Prompt engineering integrates conversation history, retrieved similar past interactions, past and current emotions, and user input into a dynamic prompt for [Ollama](https://ollama.ai)’s local LLaMA3 model.
6. **Context Management** – Short-term session memory and long-term vector-store memory ([SentenceTransformers](https://www.sbert.net/)) ensure continuity across conversations.
7. **User Interface** – A [Tkinter](https://docs.python.org/3/library/tkinter.html)-based GUI supports token-by-token streaming responses, live camera activation, and clear dialogue presentation.

The architecture is modular, allowing easy replacement or extension of emotion recognition, fusion, and response generation components.

# Installation

```bash
# Clone the repository
git clone https://git.cardiff.ac.uk/c21068582/empathetic-chatbots.git


# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

# Usage

1. Launch the GUI application.
2. Type a message into the input field — the system will record short webcam snippets during typing for facial emotion detection.
3. Emil will generate an emotionally aligned response in real time, streaming tokens into the interface.

## Personalised Classifier Training

Emil supports **user-specific emotion models** to improve facial emotion recognition accuracy.  
This feature allows the chatbot to adapt to individual expression patterns.

### Step 1: Create a User Profile
Run the script below to collect facial expression samples:

```bash
python emotion/create_user_profile.py
```

- You will be prompted for a **username**.  
- A dataset folder `user_data/<username>/` will be created with labelled images for each emotion.

### Step 2: Train the Personalised Classifier
Train a Support Vector Machine (SVM) model on your dataset:

```bash
python emotion/train_user_emotion_model.py
```

- Enter the same **username** when prompted.  
- A trained model file (`<username>.pkl`) will be saved in the appropriate directory.

### Step 3: Enable the Classifier in the Chatbot
To activate your personalised model:

1. Open `emotion/facial_emotion.py`.  
2. Locate the `VideoRec` class.  
3. Update the `user_profile` variable with your chosen **username**.

Once updated, Emil will load your personalised classifier alongside the default model during runtime.

# Testing

Before running the facial detection tests, you need to provide 7 short clips for each emotion. These clips should be saved in tests/test_videos and named after each emotion. The tests script test_emotions.py has all the paths of the videos (e.g. happy.mp4 for happy)

# Acknowledgements

We thank [DeepFace](https://github.com/serengil/deepface), [HuggingFace](https://huggingface.co), [MediaPipe](https://developers.google.com/mediapipe), and the [LangChain](https://www.langchain.com/) community for providing the core tools that made this project possible.


