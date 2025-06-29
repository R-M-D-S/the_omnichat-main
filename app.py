import streamlit as st
import os
import numpy as np
import re
import base64
import asyncio
import websockets
import json
from io import BytesIO
from dotenv import load_dotenv
from pydub import AudioSegment
import fitz  # PyMuPDF for PDF text and image extraction
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from audio_recorder_streamlit import audio_recorder
from audiorecorder import audiorecorder
import io
import wave
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.llms import OpenAI as LangOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load environment variables
load_dotenv()

# Directories for temp media
os.makedirs("slides", exist_ok=True)
os.makedirs("audio", exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Audio configuration
AUDIO_FORMAT = "pcm16"  # 16-bit PCM format for WebRTC
CHANNELS = 1
RATE = 23000
CHUNK = 1024

@tool
def calculate_area(input: str) -> str:
    """
    Calculate area of a rectangle. Input should be 'length,width'.
    Example: '5,3'
    """
    try:
        length, width = map(float, input.split(","))
        return f"The area is {length * width}"
    except Exception as e:
        return f"Error parsing input: {e}. Please provide two numbers separated by a comma."

def describe_image(image_file):
    return "An educational image related to the topic."

def generate_explanation_script(prompt, image_desc=None):
    base_prompt = f"""Generate an educational video script for the following topic: '{prompt}'.
    Structure the script into 3-5 slides. Each slide should have:
    - A title
    - A short explanation (2-4 lines)
    - Optionally refer to this image content: {image_desc if image_desc else 'No image provided'}.
    Output as a JSON list of slides with keys 'title' and 'text'."""
    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {"role": "system", "content": "You are a tutor creating educational video scripts."},
            {"role": "user", "content": base_prompt}
        ],
        temperature=0.3
    )
    return json.loads(response.choices[0].message.content)

def create_slide_image(title, text, output_path):


    def llm_generate_matplotlib_code(topic_text):
        prompt = f"""
        You are a Python data visualization assistant.
        Based on the educational topic below, write a simple and clear matplotlib plot code snippet that illustrates the concept. Avoid file output commands like savefig or show.

        Topic: {topic_text}

        Only respond with valid Python code using matplotlib (and optionally numpy). Do not include explanations.
        """
        response = client.chat.completions.create(
            model="chatgpt-4o-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content

    def generate_diagram_with_llm(topic_text):
        code = llm_generate_matplotlib_code(topic_text)
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        local_vars = {"plt": plt, "fig": fig, "ax": ax, "np": __import__('numpy')}
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            #ax.text(0.5, 0.5, f"Error:\n{str(e)}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.axis('off')
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        img = Image.frombuffer('RGBA', (width, height), buf, 'raw', 'RGBA', 0, 1)

        plt.close(fig)
        return img
    

    img = Image.new('RGB', (1280, 720), color='white')
    draw = ImageDraw.Draw(img)
    font_title = ImageFont.truetype("arialbd.ttf", 48) if os.name == 'nt' else ImageFont.load_default()
    font_text = ImageFont.truetype("arial.ttf", 32) if os.name == 'nt' else ImageFont.load_default()

    draw.text((50, 50), title, fill='black', font=font_title)

    max_width = 850
    lines = []
    words = text.split()
    line = ""

    for word in words:
        test_line = f"{line} {word}".strip()
        bbox = font_text.getbbox(test_line)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    lines.append(line)

    y_text = 150
    for line in lines:
        draw.text((50, y_text), line, font=font_text, fill='black')
        y_text += 40

    diagram_img = generate_diagram_with_llm(title + " " + text)
    img.paste(diagram_img.resize((300, 300)), (900, 50))

    img.save(output_path)



def generate_narration_audio(text, filename):
    audio = client.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        input=text
    )
    with open(filename, 'wb') as f:
        f.write(audio.content)

def assemble_video(slides, output_path):
    clips = []
    for i, slide in enumerate(slides):
        img_path = f"slides/slide_{i}.png"
        audio_path = f"audio/audio_{i}.mp3"
        create_slide_image(slide["title"], slide["text"], img_path)
        generate_narration_audio(slide["text"], audio_path)
        audio_clip = AudioFileClip(audio_path)
        img_clip = ImageClip(img_path).set_duration(audio_clip.duration).set_audio(audio_clip)
        clips.append(img_clip)
    final_video = concatenate_videoclips(clips)
    final_video.write_videofile(output_path, fps=24)

def embed_html5_video(video_path):
    video_bytes = open(video_path, 'rb').read()
    video_base64 = base64.b64encode(video_bytes).decode()
    st.html(f"""
    <video width='1700' height="400" controls onpause="alert('Paused! Ask a question now.')">
      <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
    </video>
    """)

def run_explainer_ui():
    st.title("🎥 Interactive Video Explainer")

    prompt = st.text_input("Enter a topic to explain")
    image_file = st.file_uploader("Optional: Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if st.button("Generate Video") and prompt:
        image_desc = describe_image(image_file) if image_file else None
        with st.spinner("Generating video script and narration..."):
            slides = generate_explanation_script(prompt, image_desc)
            assemble_video(slides, "explainer_video.mp4")
        st.success("Video generated successfully!")
        embed_html5_video("explainer_video.mp4")

        st.subheader("🧠 Ask a question about the video:")
        follow_up = st.text_input("Ask something related to the explanation above")
        if follow_up:
            response = client.chat.completions.create(
                model="chatgpt-4o-latest",
                messages=[
                    {"role": "system", "content": "You are a STEM tutor helping clarify video topics."},
                    {"role": "user", "content": f"The topic was: {prompt}. {follow_up}"}
                ]
            )
            st.markdown(response.choices[0].message.content)

# Function to convert audio chunks to base64-encoded PCM
def audio_chunk_to_base64(audio_chunk):
    if isinstance(audio_chunk, bytes):  # Ensure audio_chunk is in bytes format
        return base64.b64encode(audio_chunk).decode('utf-8')
    elif isinstance(audio_chunk, str):  # Handle string case, if applicable
        audio_chunk = audio_chunk.encode('utf-8')
        return base64.b64encode(audio_chunk).decode('utf-8')
    else:
        raise TypeError("Audio chunk must be of type bytes or str.")


# Real-time WebSocket function to send and receive audio
async def connect_to_openai_websocket(audio_data):
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(url, extra_headers=headers) as ws:
        # Send session configuration to OpenAI
        session_update = {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": "nova",
                "model": "tts-1-hd",
                "instructions": "You are a real-time tutor named Thuto who is knowledgable in in the CAPS curricullum in STEM subjects. Assist students as best as you can.",
                "modalities": ["audio"],
                "temperature": 0.2,
            }
        }
        await ws.send(json.dumps(session_update))

        # Send recorded audio in chunks to WebSocket
        for audio_chunk in audio_data:
            audio_base64 = audio_chunk_to_base64(audio_chunk)
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }))

        # Send signal to stop input and trigger response
        await ws.send(json.dumps({"type": "input_audio_buffer.stop"}))

        # Receive and play response in real time
        response_audio = []
        async for message in ws:
            event = json.loads(message)

            # Check for audio data in response
            if event.get('type') == 'response.audio.delta':
                response_audio.append(base64.b64decode(event['delta']))
            
            # End streaming on receiving the end signal
            if event.get('type') == 'response.audio.done':
                break

        # Concatenate audio chunks and return
        return b"".join(response_audio)

# Function to play audio in real-time within Streamlit
def play_audio_stream(audio_data):
    if audio_data:
        audio_segment = AudioSegment.from_raw(BytesIO(audio_data), sample_width=2, frame_rate=RATE, channels=1)
        with BytesIO() as buffer:
            audio_segment.export(buffer, format="wav")
            st.audio(buffer.getvalue(), format="audio/wav",autoplay=True)


def wav_to_bytes(wav_data):
    with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
        byte_io = io.BytesIO()
        byte_io.write(wav_file.readframes(wav_file.getnframes()))
        return byte_io.getvalue()



# Function to extract text and images from a PDF
def extract_content_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text += page.get_text("text")

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img = Image.open(BytesIO(image_bytes))
            images.append((f"Page {page_num + 1} - Image {img_index + 1}", img))

    return text, images

# Streamlined function for LLM response streaming
def stream_llm_response(client, model_params):
    # Ensure text fields are strings in messages
    for message in st.session_state.messages:
        if message["content"][0]["type"] == "text":
            message["content"][0]["text"] = str(message["content"][0]["text"])  # Convert to string if not already

    
    response_message = ""
    system_role = "You are an advanced AI tutor named Thuto the Tutor, specializing in all things Mathematics and science. Provide clear, structured explanations to assist students, focusing mainly on the mathematics and science related topics and nothing else."

    prompt = [{"role": "system", "content": system_role}
    ]
    prompt += st.session_state["messages"]

    for chunk in client.chat.completions.create(
        model=model_params["model"],
        messages=prompt,
        temperature=model_params["temperature"],
        max_tokens=4096,
        stream=True,
    ):
        content = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        response_message += content
        yield content

    # Append assistant response as a string
    st.session_state.messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_message}]
    })



# Function to convert file to base64 (image handling)
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def preprocess_latex(latex_text):
# Remove display mode delimiters
#latex_text = re.sub(r'\$\$(.+?)\$\$', r'\1', latex_text, flags=re.DOTALL)
# Remove inline mode delimiters
#latex_text = re.sub(r'\$(.+?)\$', r'\1', latex_text, flags=re.DOTALL)
# Remove \( \) delimiters
    latex_text = re.sub(r'\\\((.+?)\\\)', r'\1', latex_text, flags=re.DOTALL)
# Remove \[ \] delimiters
    latex_text = re.sub(r'\\\[(.+?)\\\]', r'\1', latex_text, flags=re.DOTALL)
    return latex_text


# Function to convert file to base64 (image handling)
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

# Main Function for Streamlit
def main():
    # --- Page Config ---
    st.set_page_config(
        page_title="Thuto The Tutor",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Set your Tavily API key from https://tavily.com/
    os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY") or "your-tavily-api-key"

    from langchain_tavily import TavilySearch
    from langchain_community.tools.tavily_search import TavilySearchResults

    from langchain.tools import tool

    tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

    @tool
    def web_search(input: str) -> str:
        """Search the web using Tavily. Input should be a search query like 'latest AI news'."""
        try:
            results = tavily_search_tool(input)
            return results
        except Exception as e:
            return f"Error during search: {e}"
        
    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>Thuto The Tutor</i> 💬</h1>""")
    #st.write("Ekurhuleni Map")
    image = Image.open("cropped-webq-2048x432.png")  # Replace with your image path
    st.image(image, use_column_width=True)
    # --- Side Bar ---
    with st.sidebar:
        default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
        with st.popover("🔐 OpenAI API Key"):
            openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password")

    
    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    if openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key:
        st.write("#")
        st.warning("⬅️ Please introduce your OpenAI API Key (make sure to have funds) to continue...")

        with st.sidebar:
            st.write("#")
            st.write("#")
            st.video("EQ.mp4")
            st.write("📋[Stemperiodt Blog](https://stemperiodt.co.za/blog/)")

    else:
        client = OpenAI(api_key=openai_api_key)
                # LangChain LLM setup for ReAct Agent
        llm = LangOpenAI(openai_api_key=openai_api_key, temperature=0.3)

        tools = [web_search]  # You can add more tools

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        react_agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # simpler, better parsing fallback
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
        )



        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])

        # Side bar model options and inputs
        with st.sidebar:

            st.divider()

            model = st.selectbox("Select a model:", [
                "chatgpt-4o-latest", 
                "gpt-4-turbo", 
                "gpt-3.5-turbo-16k", 
                "gpt-4", 
                "gpt-4-32k",
            ], index=0)
            
            with st.popover("⚙️ Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            audio_response = st.toggle("Audio response", value=False)
            if audio_response:
                cols = st.columns(2)
                with cols[0]:
                    tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                with cols[1]:
                    tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )

            st.divider()
                    # --- PDF Upload ---
            st.write("### **📄 Upload a PDF for analysis (text and images):**")
            uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

            st.divider() 

            # Image Upload
            if model in ["chatgpt-4o-latest", "gpt-4-turbo"]:
                    
                st.write("### **🖼️ Add an image:**")

                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        img_type = st.session_state.uploaded_img.type if st.session_state.uploaded_img else "image/jpeg"
                        raw_img = Image.open(st.session_state.uploaded_img or st.session_state.camera_img)
                        img = get_image_base64(raw_img)
                        st.session_state.messages.append(
                            {
                                "role": "user", 
                                "content": [{
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{img_type};base64,{img}"}
                                }]
                            }
                        )

                cols_img = st.columns(2)

                with cols_img[0]:
                    with st.popover("📁 Upload"):
                        st.file_uploader(
                            "Upload an image", 
                            type=["png", "jpg", "jpeg"], 
                            accept_multiple_files=False,
                            key="uploaded_img",
                            on_change=add_image_to_messages,
                        )

                with cols_img[1]:                    
                    with st.popover("📸 Camera"):
                        activate_camera = st.checkbox("Activate camera")
                        if activate_camera:
                            st.camera_input(
                                "Take a picture", 
                                key="camera_img",
                                on_change=add_image_to_messages,
                            )

# Audio Upload
            st.write("#")
            st.write("### **🎤 Add an audio:**")

            audio_prompt = None
            if "prev_speech_hash" not in st.session_state:
                st.session_state.prev_speech_hash = None

            speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
            if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
                st.session_state.prev_speech_hash = hash(speech_input)
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=("audio.wav", speech_input),
                )

                audio_prompt = transcript.text

            st.divider()
            st.video("EQ.mp4")
            st.write("📋[Stemperiodt Blog](https://stemperiodt.co.za/blog/)")


            st.divider()
            st.subheader("🎥 Interactive Video Explainer")
            show_video_explainer = st.checkbox("Enable Video Explainer")

        # Call explainer if enabled
            if show_video_explainer:
                run_explainer_ui()

            st.divider()
            st.sidebar.write("### Real time Tutor Chat")
            audio_recorder_enabled = st.sidebar.checkbox("Enable Tutor Chat")
            # Add this code block inside the main function, where you handle audio recording

            if audio_recorder_enabled:
                st.write("### Real-time Tutor Chat")

                speech_recorded = audiorecorder("Click to record", "Click to stop recording")
                
                if speech_recorded:
                    st.success("Recording successful!")
                    st.audio(speech_recorded.export().read())

                # Sending audio to WebSocket if needed
                    audio_buffer = io.BytesIO()
                    speech_recorded.export(audio_buffer, format="wav", parameters=["-ar", str(16000)])
                    audio_array = np.frombuffer(audio_buffer.getvalue()[44:], dtype=np.int16).astype(np.float32) / 32768.0
                    # Convert back to int16
                    audio_int16 = (audio_array * 32768).astype(np.int16)

                    # Convert the int16 array to bytes
                    audio_bytes = audio_int16.tobytes()

                    response_audio = asyncio.run(connect_to_openai_websocket([audio_bytes]))
                    
                    if response_audio:
                        st.success("Response received! Playing Thuto's answer...")
                        play_audio_stream(response_audio)
        
        use_react_agent = st.sidebar.checkbox("Use ReAct Agent", value=False)          
        if prompt := st.chat_input("Hi! Ask me anything...") or audio_prompt:
            st.session_state.messages.append(
                {
                    "role": "user", 
                    "content": [{
                        "type": "text",
                        "text": f"Answer the following question: {prompt}. Instruction: Please ensure all LaTex and inline mathematical expressions are wrapped in single dollar signs ($) and all display mathematical expressions are wrapped in double dollar signs ($$)." or f"Answer the following question: {audio_prompt}. Instruction: Please ensure all LaTex and inline mathematical expressions are wrapped in single dollar signs ($) and all display mathematical expressions are wrapped in double dollar signs ($$).",
                    }]
                }
            )
            
            # Displaying the new messages
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                latex_output = ""
                if use_react_agent:
                    agent_response = react_agent.run(prompt)
                    latex_output = agent_response
                else:
                    for chunk in stream_llm_response(client, model_params):
                        latex_output += chunk

                
                def preprocess_latex(latex_text):
                    # Remove display mode delimiters
                    #latex_text = re.sub(r'\$\$(.+?)\$\$', r'\1', latex_text, flags=re.DOTALL)
                    # Remove inline mode delimiters
                    #latex_text = re.sub(r'\$(.+?)\$', r'\1', latex_text, flags=re.DOTALL)
                    # Remove \( \) delimiters
                    latex_text = re.sub(r'\\\((.+?)\\\)', r'\1', latex_text, flags=re.DOTALL)
                    # Remove \[ \] delimiters
                    latex_text = re.sub(r'\\\[(.+?)\\\]', r'\1', latex_text, flags=re.DOTALL)
                    return latex_text
                latex = preprocess_latex(latex_output)
                st.markdown(latex)

            # --- Added Audio Response (optional) ---
            if audio_response:
                response =  client.audio.speech.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=st.session_state.messages[-1]["content"][0]["text"],
                )
                audio_base64 = base64.b64encode(response.content).decode('utf-8')
                audio_html = f"""
                <audio controls autoplay>
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
                </audio>
                """
                st.html(audio_html)
  
        # Process uploaded PDF
        if uploaded_pdf:
            with st.spinner("Extracting content from the PDF..."):
                pdf_text, pdf_images = extract_content_from_pdf(uploaded_pdf)
                st.write("### Extracted text from the PDF:")
                st.write(pdf_text)

                if pdf_images:
                    st.write("### Extracted images from the PDF:")
                    for img_name, img in pdf_images:
                        st.write(img_name)
                        st.image(img)

                # Use extracted text to query the LLM
                if prompt := st.chat_input("Analyze the extracted PDF content or ask a question..."):
                    user_message = f"Analyze the following PDF content:\n\n{pdf_text}\n\nUser's question: {prompt}"
                    st.session_state.messages.append(
                        {"role": "user", "content": [{"type": "text", "text": user_message}]}
                    )

                    # Displaying the new user message
                    with st.chat_message("user"):
                        st.markdown(user_message)

                    # Query the LLM
                    with st.chat_message("assistant"):
                        latex_output = ""
                        for chunk in stream_llm_response(client, model_params):
                            latex_output += chunk
                        st.markdown(latex_output)

if __name__ == "__main__":
    # Start the Streamlit app (runs on the main thread)
    main()