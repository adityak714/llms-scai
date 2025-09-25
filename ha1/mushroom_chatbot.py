"""A mushroom expert chatbot that responds to user queries about mushrooms."""
from typing import Iterator
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

client = genai.Client()
model = "gemini-2.0-flash"

import gradio as gr
from gradio import ChatMessage
from PIL import Image

def stream_gemini_response(user_message: str, messages: list, file, temp) -> Iterator[list]:
    history.append(ChatMessage(role="user", content=user_message))
    
    instructions = "You are an assistant bot that is only to discuss about mushrooms. You have to also talk to the user in a natural fashion, so that you do not sound like a robot. Understand the question given to you, and check if it relates to mushrooms, or information about them. If it does not, tell the user to ask a new question, or to reformulate the question. Use the information made available to you, and provide an appropriate response with the resources for mushroom knowledge you have."
    
    prompt = f'{"Question: " + history[-1].content if history[-1].content != "" else ""}' + "\n\nHistory:\n" + "".join([record.content for record in history[:-1]])
    empty_message = False

    # Initially set the payload to be the prompt
    payload = prompt

    if file:
        print("Image was uploaded. Continuing....")
        # If an image was uploaded, update the value of payload to be now an 
        # array of [prompt, file (in PIL Image form)] 
        payload = [prompt, Image.open(file)]
    if file and user_message == "":
        empty_message = True
        instructions = """You are an assistant bot that is only to discuss about mushrooms. If the question is absent, and just an image is present, you must generate a valid JSON object as your response, containing the attributes: 
        
        {
            common_name, 
            genus, 
            confidence (of your prediction), 
            visible (what parts of the mushrooms are visible in the image, selecting one, multiple OR all of the enumerations {cap, hymenium, stipe}), 
            color (of the mushroom in the picture),
            edible (is the mushroom edible? must be a {boolean})
        }

        Therefore, you must have 6 attributes in your generated JSON, and only that is your answer format. Strict JSON syntax must be followed."""

    config = types.GenerateContentConfig(
                system_instruction=instructions,
                temperature=temp,
                # Safety feedback -- generateContent returns a GenerateContentResponse which includes safety feedback.
                safety_settings=[
                    types.SafetySetting(
                        category='HARM_CATEGORY_DANGEROUS_CONTENT',
                        threshold='BLOCK_LOW_AND_ABOVE',
                    )
                ]
            )

    if not empty_message: 
        # Initialize buffers
        thought_buffer = ""
        response_buffer = ""
        thinking_complete = False
        
        # Add initial thinking message
        history.append(
            ChatMessage(
                role="assistant",
                content=""
            )
        )

        for chunk in client.models.generate_content_stream(model=model, contents=payload, config=config):
            try:
                parts = chunk.candidates[0].content.parts
                current_chunk = parts[0].text

                if len(parts) == 2 and not thinking_complete:
                    # Complete thought and start response
                    thought_buffer += current_chunk
                    history[-1] = ChatMessage(
                        role="assistant",
                        content=thought_buffer
                    )
                    
                    # Add response message
                    history.append(
                        ChatMessage(
                            role="assistant",
                            content=parts[1].text
                        )
                    )
                    thinking_complete = True
                    
                elif thinking_complete:
                    # Continue streaming response
                    response_buffer += current_chunk
                    history[-1] = ChatMessage(
                        role="assistant",
                        content=response_buffer
                    )
                    
                else:
                    # Continue streaming thoughts
                    thought_buffer += current_chunk
                    history[-1] = ChatMessage(
                        role="assistant",
                        content=thought_buffer
                    )

                yield history[-1]
            except Exception:
                if chunk.usage_metadata:
                    print("\n>>>>>>>>>>>>>>>>>>> Safety Filter Triggered <<<<<<<<<<<<<<<<<<<<<<<")
                    print(chunk) # Print what safety filters were violated in just the terminal
                    print(">>>>>>>>>>>>>>>>>>> *********************** <<<<<<<<<<<<<<<<<<<<<<<")
                yield "There was an error. Please try again."
    else:
        # LLM still records it in the array of messages, and 
        # does not reveal to the user.
        new_client = genai.Client()
        response = new_client.models.generate_content(model=model, contents=payload, config=config)

        # Print JSON to the console
        print(response.text
            .replace("```json", "")   # remove markdown wrappings generated by the LLM
            .replace("```", ""))      # as they are unnecessary.
        
        history.append(        
            ChatMessage(
                role="assistant", 
                content=response.text
            )
        )

        yield "Noted. Ask away about it! \n" + response.text
            

#########################################
with gr.Blocks(theme=gr.themes.Ocean(), fill_height=True) as demo:
    history = []

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            image = gr.Image(type="filepath", height=300, sources=["upload", "clipboard"])    
            question = gr.Textbox(placeholder="Type your message here and press Enter...")
            # slider = gr.Slider(0.0, 1.0, step=0.05, label="Temperature")
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(type="messages", autoscroll=True)
            chat_interface = gr.ChatInterface(
                stream_gemini_response,
                type="messages",
                title="Mushroom Chatbot - your go-to for all mushroom-related queries!",
                multimodal=True,
                chatbot=chatbot,
                textbox=question,
                additional_inputs=[
                    image, 
                    gr.Slider(0.0, 1.0, step=0.05, label="Temperature", value=0.7)
                ]
            )

#########################################

if __name__ == "__main__":
    demo.launch(debug=True)
