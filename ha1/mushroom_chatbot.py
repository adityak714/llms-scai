"""A mushroom expert chatbot that responds to user queries about mushrooms."""
from typing import Iterator
from dotenv import load_dotenv
load_dotenv()

import os, random

API_KEY = str(os.getenv("GEMINI_API_KEY"))

import google.generativeai as genai

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

import gradio as gr
from gradio import ChatMessage
import base64
from PIL import Image

def image_to_base64(image_path):
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')

def stream_gemini_response(file, user_message: str, messages: list) -> Iterator[list]:
    """    Streams both thoughts and responses from the Gemini model.    """
    payload = user_message

    if file:
        print("Image was uploaded. Continuing....")
        base64img = image_to_base64(file)
        print(payload + " " + f'data:image/jpeg;base64,{base64img}')
        payload = [user_message, Image.open(file)]

    print(user_message, messages)

    # Initialize response from Gemini
    response = model.generate_content(payload, stream=True)
    
    # Initialize buffers
    thought_buffer = ""
    response_buffer = ""
    thinking_complete = False
    
    # Add initial thinking message
    messages.append(
        ChatMessage(
            role="assistant",
            content="",
        )
    )
    
    for chunk in response:
        parts = chunk.candidates[0].content.parts
        current_chunk = parts[0].text
        
        if len(parts) == 2 and not thinking_complete:
            # Complete thought and start response
            thought_buffer += current_chunk
            messages[-1] = ChatMessage(
                role="assistant",
                content=thought_buffer,
                metadata={"title": "Thinking: *The thoughts produced by the Gemini2.0 Flash model are experimental"}
            )
            
            # Add response message
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=parts[1].text
                )
            )
            thinking_complete = True
            
        elif thinking_complete:
            # Continue streaming response
            response_buffer += current_chunk
            messages[-1] = ChatMessage(
                role="assistant",
                content=response_buffer
            )
            
        else:
            # Continue streaming thoughts
            thought_buffer += current_chunk
            messages[-1] = ChatMessage(
                role="assistant",
                content=thought_buffer,
                metadata={"title": "Thinking: *The thoughts produced by the Gemini2.0 Flash model are experimental"}
            )
        
        yield messages

#########################################
with gr.Blocks(theme=gr.themes.Ocean(), fill_height=True) as demo:
    """        title=Mushroom Chatbot, 
                description=For all your mushroom-related queries!
    """
    chatbot = gr.Chatbot(type="messages")
    image = gr.Image(type="filepath")
    question = gr.Textbox(placeholder="Type your message here and press Enter...")

    send_button = gr.Button("Send")
    send_req = send_button.click(stream_gemini_response, [image, question, chatbot], [chatbot])

#########################################

if __name__ == "__main__":
    demo.launch(debug=True)
