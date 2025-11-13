import tkinter as tk
from tkinter import scrolledtext
import threading


from langchain_ollama import OllamaLLM # import the model
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from emotion.facial_emotion import VideoRec
from emotion.text_emotion import TextEmotionDetector
from emotion.fusion import fuse_emotions
from context.memory import *
import time
import os

#Handles calling the Ollama model

# creating a template for each prompt to be able to store messages as a history
#chatbot can use the history as context for the conversation
template = """
You are an empathetic chatbot. Always tailor your responses to reflect the user's emotional state 

Here is the conversation history: {context}

User Emotion: {emotion}

User Message: {user_query}

Response: 
"""
#setting up the model that will be used for the response generation
model = OllamaLLM(
    model="llama3",
    streaming=True,  # <--- Enable streaming
    callbacks=[StreamingStdOutCallbackHandler()]
)
prompt = ChatPromptTemplate.from_template(template) #initialing the template for the prompt
chain  = prompt | model #linking the prompt template to the model

#procedure to handle the chat conversation
def conversation():
    text_emotion_detector = TextEmotionDetector()
    context = ""
    print("Welcome to the Empathic Chatbot! My name is Emil. How are you feeling today?")
    print("You can type 'exit' or 'bye' to leave the conversation.")

    #basic chat loop
    while True:
        recorder = VideoRec() #initialise the video capture object
        print()
        try:
            recorder.start() #start recording
            userinput = input("You: ") #get the user message
            #checks if the user wants to end the chat and closes the conversation
            if userinput.lower() in ["bye", "exit", "quit", "adios"]:
                print("See you again next time!")
                recorder.stop()
                os.remove(recorder.filename)
                print()
                print(str(recorder.filename), " is deleted!")
                overall_emotion = get_overall_emotion()
                log_conversation_mood(overall_emotion)
                reset_session()
                break
            recorder.stop() #stops the recording

            #calls the fucntion to process the video and extract the emotion
            facial_emotion = recorder.extract_emotion(recorder.filename)
            print(f"Facial emotion: {facial_emotion}") #prints emotion for testing purposes

            text_emotion = text_emotion_detector.detect_emotion(userinput)
            print(f"Text emotion: {text_emotion}")

            # Fuse both
            final_emotion = fuse_emotions(facial_emotion, text_emotion)
            print(f"Fused emotion: {final_emotion}")

            #logging user message for context
            log_message("user", userinput, final_emotion)
            add_to_vector_store(userinput, "user")

            
        except Exception as e: #for catching errors during recording
            print("Error during recording:", e)
            break
        #gets the context to be used for the chatbot response
        context = get_formatted_context()
        print(context)
        #chatbot response
        print()
        print("\nEmil: ", end="")
        emotion = "lonely" #sets the emotion for testing purposes
        #invokes the model to generate the chatbot response
        #this also makes sure the response is being printed as it is generating
        result = chain.invoke({"context": context, "emotion": final_emotion, "user_query": userinput})
            #print("Emil: ", result)
        #early context management variable - placeholder for now
        #context += f"\nUser: {userinput}\nEmil(AI): {result}"
        log_message("bot", result, "neutral")
        add_to_vector_store(result, "bot")
        print()
        os.remove(recorder.filename) #deletes the recording after it has been processed
        print(str(recorder.filename), " is deleted!")
        
         