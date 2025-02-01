#!/usr/bin/env python
# coding: utf-8

"""
This script handles sound file analysis, audio feature extraction, and machine learning-based
music genre classification. It includes visualization, preprocessing, and a trained model for prediction.
"""

# Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Sequential
from tensorflow.keras.models import load_model
import IPython.display as ipd

# Visualization and Feature Extraction Example
def visualize_audio(audio_path):
    """
    Visualizes waveform, spectrogram, and other features of an audio file.

    Parameters:
        audio_path (str): Path to the audio file.
    """
    # Load the audio file
    data, sr = librosa.load(audio_path)
    print(f"Audio Type: {type(data)}, Sample Rate Type: {type(sr)}")

    # Plot waveform
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(data, sr=sr, color="#2B4F72")
    plt.title("Waveform")
    plt.show()

    # Plot spectrogram (log scale)
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.title("Spectrogram (Log Scale)")
    plt.show()

# Feature Extraction
def extract_features(file_path):
    """
    Extracts audio features from the given file to match the training dataset.

    Parameters:
        file_path (str): Path to the audio file.

    Returns:
        numpy.ndarray: Scaled features (same as training input shape).
    """
    # Load the audio file
    y, sr = librosa.load(file_path, mono=True, duration=30)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Aggregate features (e.g., mean values for each feature set)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    chroma_scaled = np.mean(chroma.T, axis=0)
    spectral_contrast_scaled = np.mean(spectral_contrast.T, axis=0)[:-1]  # Exclude last feature to make 58 total

    # Combine all features into a single numpy array
    features = np.hstack([mfccs_scaled, chroma_scaled, spectral_contrast_scaled])

    # Debugging step: Print the feature shape
    print(f"Extracted features shape: {features.shape}")

    return features

# Genre Prediction
def predict_genre(file_path):
    """
    Predicts the genre of a given audio file using a pre-trained model.
    """
    model = load_model('music_genre_classifier.keras')

    # Extract features
    features = extract_features(file_path)

    # Check feature dimensions
    if features.shape[0] != 58:
        raise ValueError(f"Feature shape mismatch: Expected 58, got {features.shape[0]}")

    # Reshape for model input
    features = features.reshape(1, -1)

    # Predict genre
    prediction = model.predict(features)
    predicted_genre = np.argmax(prediction, axis=1)

    genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop',
                    'jazz', 'metal', 'pop', 'reggae', 'rock']
    return genre_labels[predicted_genre[0]]

# Main Block for Testing and Visualization
if __name__ == '__main__':
    # File paths
    audio_path = r"C:\Users\siddh\OneDrive\Desktop\Genre\Datsaset\classical\classical.00096.wav"
    dataset_path = r"C:\Users\siddh\OneDrive\Desktop\Genre\Datsaset\features_3_sec.csv"

    # Debugging: Check feature shape
    print("Testing feature extraction for debugging...")
    test_features = extract_features(audio_path)
    print(f"Extracted Feature Shape: {test_features.shape} (Expected: 58)")

    # Visualization Example
    print("Visualizing Audio Features...")
    visualize_audio(audio_path)

  

    # Test prediction
    print("Predicting genre...")
    genre = predict_genre(audio_path)
    print(f"Predicted Genre: {genre}")
