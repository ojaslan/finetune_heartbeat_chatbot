import streamlit as st
import tempfile
from model.qwen_audio_chat import QwenAudioChat
from utils.audio_utils import preprocess_audio

st.set_page_config(page_title="Heartbeat LLM Chatbot", layout="centered")
st.title("🫀 Heartbeat LLM Chatbot")

@st.cache_resource
def load_model():
    return QwenAudioChat()

chatbot = load_model()

uploaded_audio = st.file_uploader("Upload heartbeat audio (.wav)", type=["wav"])
user_question = st.text_input("Ask something about the heartbeat")

if uploaded_audio and user_question:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(uploaded_audio.read())
        raw_path = temp.name

    processed_path = preprocess_audio(raw_path)

    with st.spinner("Analyzing heartbeat with LLM..."):
        response = chatbot.chat(processed_path, user_question)

    st.markdown("### 🤖 AI Response")
    st.success(response)
