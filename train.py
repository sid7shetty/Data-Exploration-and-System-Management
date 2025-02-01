import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\siddh\OneDrive\Desktop\Genre\Datsaset\blues\blues.00000.wav"


def extract_mel_spectrogram(file_path, sr=22050, n_mels=128, hop_length=512):
    y, sr = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def plot_spectrogram(mel_spec_db):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=22050)
    plt.colorbar(format='%+2.0f dB')
    plt.show()
