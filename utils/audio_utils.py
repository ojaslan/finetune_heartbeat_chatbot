import librosa
import soundfile as sf
import numpy as np

TARGET_SR = 16000

def preprocess_audio(input_path, output_path="processed.wav"):
    audio, sr = librosa.load(input_path, sr=None)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    audio = audio / np.max(np.abs(audio))  # normalize

    sf.write(output_path, audio, TARGET_SR)
    return output_path
