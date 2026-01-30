import requests
import base64
import os

HF_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2-Audio-7B-Instruct"

HF_TOKEN = os.getenv("HF_TOKEN")  # set this in Streamlit secrets

class QwenAudioChat:
    def __init__(self):
        if HF_TOKEN is None:
            raise ValueError("HF_TOKEN not set in environment variables")

        self.headers = {
            "Authorization": f"Bearer {HF_TOKEN}"
        }

    def chat(self, audio_path, question):
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "inputs": {
                "audio": audio_base64,
                "text": question
            }
        }

        response = requests.post(
            HF_API_URL,
            headers=self.headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            return f"API Error: {response.text}"

        result = response.json()

        if isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]

        return str(result)
