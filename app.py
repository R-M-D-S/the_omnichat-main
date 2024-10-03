import streamlit as st
from openai import OpenAI
import dotenv
import os
import fitz  # PyMuPDF for PDF text and image extraction
from PIL import Image
from io import BytesIO
import base64

dotenv.load_dotenv()

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

# Function to query and stream the response from the LLM
def stream_llm_response(client, model_params):
    response_message = ""

    for chunk in client.chat.completions.create(
        model=model_params["model"] if "model" in model_params else "gpt-4",
        messages=st.session_state.messages,
        temperature=model_params["temperature"] if "temperature" in model_params else 0.0,
        max_tokens=4096,
        stream=True,
    ):
        response_message += chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
        yield chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""

    st.session_state.messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})

# Function to convert file to base64 (image handling)
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

def main():

    # --- Page Config ---
    st.set_page_config(
        page_title="Thuto The Tutor",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.markdown("<h1 style='text-align: center; color: #6ca395;'>🤖 <i>Thuto The Tutor</i> 💬</h1>", unsafe_allow_html=True)
    image = Image.open("cropped-webq-2048x432.png")  # Replace with your image path
    st.image(image, use_column_width=True)

    # --- Side Bar ---
    with st.sidebar:
        default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""
        with st.expander("🔐 OpenAI API Key"):
            openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password")

        st.divider()

        model = st.selectbox("Select a model:", [
            "chatgpt-4o-latest", 
            "gpt-4-turbo", 
            "gpt-3.5-turbo-16k", 
            "gpt-4", 
            "gpt-4-32k",
        ], index=0)

        with st.expander("⚙️ Model parameters"):
            model_temp = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)

        audio_response = st.checkbox("Audio response", value=False)
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

        st.button("🗑️ Reset conversation", on_click=reset_conversation)

        st.divider()

        # --- PDF Upload ---
        st.write("### **📄 Upload a PDF for analysis (text and images):**")
        uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

        st.divider()

        # --- Image Upload and Camera ---
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
            with st.expander("📁 Upload"):
                st.file_uploader(
                    "Upload an image", 
                    type=["png", "jpg", "jpeg"], 
                    accept_multiple_files=False,
                    key="uploaded_img",
                    on_change=add_image_to_messages,
                )

        with cols_img[1]:                    
            with st.expander("📸 Camera"):
                activate_camera = st.checkbox("Activate camera")
                if activate_camera:
                    st.camera_input(
                        "Take a picture", 
                        key="camera_img",
                        on_change=add_image_to_messages,
                    )

        st.divider()

        st.video("EQ.mp4")
        st.write("📋[Stemperiodt Blog](https://stemperiodt.co.za/blog/)")

    # --- Main Content ---
    if openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key:
        st.write("#")
        st.warning("⬅️ Please introduce your OpenAI API Key (make sure to have funds) to continue...")

        with st.sidebar:
            st.write("#")
            st.video("EQ.mp4")
            st.write("📋[Stemperiodt Blog](https://stemperiodt.co.za/blog/)")
    
    else:
        client = OpenAI(api_key=openai_api_key)

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

        else:
            # Allow user to input a message without uploading a PDF
            if prompt := st.chat_input("Type your message here..."):
                user_message = prompt
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

        # Audio Response (optional)
        if audio_response and st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            response_text = st.session_state.messages[-1]["content"][0]["text"]
            response = client.audio.speech.create(
                model=tts_model,
                voice=tts_voice,
                input=response_text,
            )
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            audio_html = f"""
            <audio controls autoplay>
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
