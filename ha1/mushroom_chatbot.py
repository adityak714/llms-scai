"""A mushroom expert chatbot that responds to user queries about mushrooms."""
import gradio as gr, random
from gradio import ChatMessage

def response(message, history):
    if len(message["files"]) != 0:
        return "A file seems to have been uploaded."
    if "-test-" in message["text"].lower():
        return "The interface is up. However, needs to be connected to some strong LLM."
    if "hello" in message["text"].lower():
        return "Hello! I am a mushroom expert. Ask me anything about mushrooms."
    elif "bye" in message["text"].lower():
        return "Goodbye! Have a great day."
    else:
        return random.choice([
            "I am not sure about that. Can you ask me something else?",
            "Could you reformulate your question? I am not sure I understand.",
            "I don't understand, can you ask someone else?",
            "What a stellar question! I am not sure about the answer though.",
            "You know what? I am not cut out for this. I am going to take a break."
        ])

#######################################
# TODO: Add the Gemini (or HuggingFace) chatbot to the interface.
from typing import Iterator
from dotenv import load_dotenv
import os, google.generativeai as genai

load_dotenv()

API_KEY = str(os.getenv("GEMINI_API_KEY"))

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def stream_gemini_response(user_message: str, messages: list) -> Iterator[list]:
    """
    Streams both thoughts and responses from the Gemini model.
    """
    # Initialize response from Gemini
    response = model.generate_content(user_message, stream=True)
    
    # Initialize buffers
    thought_buffer = ""
    response_buffer = ""
    thinking_complete = False
    
    # Add initial thinking message
    messages.append(
        ChatMessage(
            role="assistant",
            content="",
            metadata={"title": "⏳ Thinking: *The thoughts produced by the Gemini 2.0 Flash model are experimental*"}
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
                metadata={"title": "⏳Thinking: *The thoughts produced by the Gemini2.0 Flash model are experimental"}
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
                metadata={"title": "⏳Thinking: *The thoughts produced by the Gemini2.0 Flash model are experimental"}
            )
        
        yield messages

#########################################
with gr.Blocks(fill_height=True) as demo:
    # TODO: ADD A SYSTEM PROMPT + HISTORY
    
    # TODO: -->
    # system_prompt = gr.TextBox("You are a mushroom expert chatbot that responds to user queries about mushrooms.", label="System Prompt")

    chatbot = gr.Chatbot(
        #fn=response,
        type="messages",
        # inputs=["text", "image"],
        # outputs=["text"],
        # textbox=gr.MultimodalTextbox(
        #     file_count='single',       
        #     file_types=[".png", ".jpg", ".jpeg", "image"]
        # ),
        label="Your Personal Mushroom Expert",
        render_markdown=True
    )

    input_box = gr.Textbox(
        lines=1,
        label="Chat Message",
        placeholder="Type your message here and press Enter..."
    )

    input_box.submit(stream_gemini_response, [input_box, chatbot], [chatbot])
#########################################

if __name__ == "__main__":
    demo.launch()
