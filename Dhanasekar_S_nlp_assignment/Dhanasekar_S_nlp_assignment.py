import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to load audio data and transcriptions
def load_data(file_path):
    audio_data = []
    transcriptions = []

    audio_files = [file for file in os.listdir(file_path) if file.endswith('.wav')]
    transcription_file = [file for file in os.listdir(file_path) if file.endswith('.txt')]

    if len(transcription_file) != 1:
        raise ValueError("There should be exactly one .txt file containing transcriptions.")

    with open(os.path.join(file_path, transcription_file[0]), 'r', encoding='utf-8') as file:
        transcription_content = file.read().splitlines()

    if len(transcription_content) != len(audio_files):
        raise ValueError("Number of transcriptions does not match the number of audio files.")

    for audio_file, transcription in zip(audio_files, transcription_content):
        audio_path = os.path.join(file_path, audio_file)
        audio, sr = librosa.load(audio_path, sr=16000)  
        audio_data.append(audio)
        transcriptions.append(transcription.strip())

    return audio_data, transcriptions

# Convert transcriptions to numerical labels using LabelEncoder
def preprocess_audio(audio):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=2048, hop_length=512)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    desired_shape = (128, 128)
    if spectrogram.shape[1] < desired_shape[1]:  # Padding
        pad_width = desired_shape[1] - spectrogram.shape[1]
        spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    elif spectrogram.shape[1] > desired_shape[1]:  # Truncating
        spectrogram = spectrogram[:, :desired_shape[1]]
    return spectrogram

# Load data using the defined function
file_path = r"C:/Users/Protectt067/OneDrive - PROTECTT AI LABS PVT LTD/Documents/common_voice_test"
audio_data, transcriptions = load_data(file_path)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(transcriptions)
X_train, X_test, y_train, y_test = train_test_split(audio_data, y, test_size=0.2, random_state=42)

# Preprocess the audio data
X_train_processed = [preprocess_audio(audio) for audio in X_train]
X_test_processed = [preprocess_audio(audio) for audio in X_test]

X_train_processed = np.array(X_train_processed)
X_test_processed = np.array(X_test_processed)

# Create model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128)),
    tf.keras.layers.Reshape((128, 128, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile and fit the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_processed, y_train, validation_data=(X_test_processed, y_test), epochs=10, batch_size=32)

# Transcription function
def get_transcription(filename):
    audio, _ = librosa.load(filename, sr=16000)
    processed_audio = preprocess_audio(audio)
    processed_audio = np.expand_dims(processed_audio, axis=0)

    prediction = model.predict(processed_audio)
    predicted_label = np.argmax(prediction)
    
    transcription = label_encoder.inverse_transform([predicted_label])[0]
    return transcription

# get_transcription function
sample_audio_file = r"C:/Users/Protectt067/OneDrive - PROTECTT AI LABS PVT LTD/Documents/common_voice_test/common_voice_mr_27591986.wav"
predicted_transcription = get_transcription(sample_audio_file)
print(f"Predicted Transcription: {predicted_transcription}")