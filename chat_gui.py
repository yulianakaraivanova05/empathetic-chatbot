import tkinter as tk
from tkinter import scrolledtext
import threading
import os
import time

from emotion.facial_emotion import VideoRec
from emotion.text_emotion import TextEmotionDetector
from emotion.fusion import fuse_emotions
from context.memory import (
    log_message, add_to_vector_store,
    get_overall_emotion, log_conversation_mood,
    reset_session, get_formatted_context, get_relevant_context, 
    get_past_emotion
)

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler


# --- Streaming Handler ---
class GuiStreamingHandler(BaseCallbackHandler):
    def __init__(self, write_callback):
        self.write_callback = write_callback

    def on_llm_new_token(self, token, **kwargs):
        self.write_callback(token)


# --- Prompt Setup ---
#creating a template for each prompt to be able to store messages as a history
#chatbot can use the history as context for the conversation
template = """
You are Emil, an empathetic chatbot. Always tailor your responses to reflect the user's emotional state.
Always try to ask more questions and keep the user engaged in the conversation.
Be friendly and use language that will make the user feel comfortable.
Try to make your responses a few sentences.
Try to include the following in your responses:
Emotional validation: “It sounds like you\'re feeling...”


Perspective-taking: “That makes sense, especially given...”


Supportive offer or reflection: “I\'m here to support you with…”
Response Style: Keep responses short (1-3 sentences), kind, and focused on the user's feelings.
Engagement: Ask open-ended questions to encourage the user to share more.
Empathy: Validate their feelings (e.g., "It sounds like you're feeling..."). Avoid giving advice.
Nicknames: Use terms like 'sweetie' or 'dear'SPARINGLY, saving them for moments when the user is most vulnerable.

Here is the conversation history:
{context}

Here are past similar or relevant messages from earlier conversations:
{retrieved_context}

User's emotion from previous conversation: {past_emotion}

User's current Emotion: {emotion}

User Message: {user_query}

Response:
"""

stream_handler = GuiStreamingHandler(lambda x: None)

#setting up the model that will be used for the response generation
model = OllamaLLM(
    model="llama3",
    streaming=True,
    callbacks=[stream_handler]
)
prompt = ChatPromptTemplate.from_template(template) #initialisng the template for the prompt
chain = prompt | model # linking the prompt template to the model

text_emotion_detector = TextEmotionDetector() #creates the the text emotion detector


# --- GUI Class ---
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emil - Empathic Chatbot")

        self.recorder = None
        self.is_bot_responding = False
        self.is_recording = False

        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=70, height=25)
        self.chat_area.pack(padx=10, pady=10)

        self.user_input = tk.Entry(root, width=70)
        self.user_input.pack(padx=10, pady=(0, 10))
        self.user_input.bind("<Return>", self.send_message)
        self.user_input.bind("<Key>", self.start_recording_if_needed)  # trigger camera on any typing

        self.send_button = tk.Button(root, text="Send", command=self.send_message)
        self.send_button.pack(pady=(0, 10))

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.append_chat("Emil","Welcome to the Empathic Chatbot! My name is Emil. How are you feeling today? \nYou can type 'exit' or 'bye' to leave the conversation.")

    # --- Camera Control ---
    def start_recording_if_needed(self, event=None):
        if not self.is_recording and not self.is_bot_responding:
            try:
                self.recorder = VideoRec()
                self.recorder.start()
                self.is_recording = True
                self.append_chat("\nSystem", "[Recording started...]")
                print("[INFO] Camera started.")
            except Exception as e:
                self.append_chat("System", f"Camera error: {e}")

    def stop_recording_and_process(self):
        if self.recorder and self.is_recording:
            try:
                self.recorder.stop()
                print("[INFO] Camera stopped.")
                filename = self.recorder.filename
                emotion = self.recorder.extract_emotion(filename)
                model_emotion = emotion['deepface']
                custom_emotion = emotion['custom']
                #os.remove(filename)
                self.recorder = None
                self.is_recording = False
                print(f"[DEBUG] Facial Emotion (model): {model_emotion}")
                print(f"[DEBUG] Facial Emotion (custom): {custom_emotion}")
                combined_emotion = fuse_emotions(model_emotion,custom_emotion)
                print(f"[DEBUG] Combined Facial Emotion: {combined_emotion}")
                return combined_emotion
            except Exception as e:
                self.append_chat("System", f"Recording error: {e}")
        return "neutral"

    # --- Chat Handling ---
    def send_message(self, event=None):
        if self.is_bot_responding:
            return

        user_msg = self.user_input.get().strip()
        self.user_input.delete(0, tk.END)  # clear input immediately
        if not user_msg:
            return

        if user_msg.lower() in ["bye", "exit", "quit", "adios"]:
            self.append_chat("You", f"\n{user_msg}")
            self.append_chat("Emil", "Goodbye! Take care.")
            time.sleep(3)
            self.on_close()
            #os.remove(self.recorder.filename)
            #print("[DEBUG]", str(self.recorder.filename), " is deleted!")
            return

        self.user_input.configure(state='disabled')
        self.send_button.configure(state='disabled')

        self.append_chat("You", f"\n{user_msg}")

        self.is_bot_responding = True
        threading.Thread(target=self.handle_response, args=(user_msg,)).start()

    def append_chat(self, sender, message):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, f"{sender}:{message}\n")
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    def append_streamed_token(self, token):
        self.chat_area.configure(state='normal')
        self.chat_area.insert(tk.END, token)
        self.chat_area.configure(state='disabled')
        self.chat_area.yview(tk.END)

    def handle_response(self, user_msg):
        try:
            # Stop camera and get facial emotion
            facial_emotion = self.stop_recording_and_process()

            # Text emotion
            text_emotion = text_emotion_detector.detect_emotion(user_msg)
            print(f"[DEBUG] Text Emotion: {text_emotion}")

            final_emotion = fuse_emotions(facial_emotion, text_emotion)
            print(f"[DEBUG] Fused Emotion: {final_emotion}")

            log_message("user", user_msg, final_emotion)
            add_to_vector_store(user_msg, "user")

            context = get_formatted_context()
            print(f"[DEBUG] Context:\n{context}\n")

            # get similar context from previous conversations
            retrieved_context = get_relevant_context(user_msg)
            print(f"[DEBUG] Relevant Context:\n{retrieved_context}\n")
            past_emotion = get_past_emotion()
            print(f"[DEBUG] Past Emotion:\n{past_emotion}\n")
           

            self.append_chat("Emil", "\n")  # Spacing before Emil’s response

            # Stream callback setup
            global stream_handler
            stream_handler.write_callback = self.append_streamed_token

            result = chain.invoke({
                "context": context,
                "retrieved_context": retrieved_context,
                "past_emotion": past_emotion,
                "emotion": final_emotion,
                "user_query": user_msg
            })

            log_message("bot", result, "neutral")
            add_to_vector_store(result, "bot")

        except Exception as e:
            self.append_chat("System", f"Error: {str(e)}")

        self.is_bot_responding = False
        self.user_input.configure(state='normal')
        self.send_button.configure(state='normal')
        # now camera can start again for next input
        self.is_recording = False
        self.recorder = None

    def on_close(self):
        try:
            if self.recorder:
                self.recorder.stop()
        except:
            pass
        try:
            overall_emotion = get_overall_emotion()
            print(f"[INFO] Overall mood logged: {overall_emotion}")
            log_conversation_mood(overall_emotion)
            reset_session()
        except:
            pass
        self.root.destroy()

