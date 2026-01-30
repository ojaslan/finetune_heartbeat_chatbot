import librosa
import soundfile as sf

def preprocess_audio(input_path):
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    output_path = input_path.replace(".wav", "_processed.wav")
    sf.write(output_path, y, 16000)
    return output_path
