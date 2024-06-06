import time
import os
import logging
import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
from src.model import CNN  # Assuming CNN model definition
from setup_logging import setup_logging
from settings import CLASSES

import pyaudio
import wave

# Replacing the sound module with the record function
def record(max_duration):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = max_duration
    WAVE_OUTPUT_FILENAME = WAVE_OUTPUT_FILE

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    print("* Recording...")

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Done recording!")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

setup_logging()
logger = logging.getLogger('app')


def init_model():
    cnn = CNN((128, 87))
    cnn.load_model()
    return cnn


def get_spectrogram(audio_file, type='mel'):
    logger.info("Extracting spectrogram")
    y, sr = librosa.load(audio_file, duration=None)  # Load entire audio
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logger.info("Spectrogram Extracted")
    format = '%+2.0f'
    if type == 'DB':
        ps = librosa.power_to_db(ps, ref=np.max)
        format = ''.join([format, 'DB'])
        logger.info("Converted to DB scale")
    return ps, format


def display(spectrogram, format):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')
    plt.title('Mel-frequency spectrogram')
    plt.colorbar(format=format)
    plt.tight_layout()
    st.pyplot(clear_figure=False)


def main():
    title = "Musical Chord Recognition"
    st.title(title)
    image = Image.open(os.path.join(IMAGE_DIR, 'app_guitar.jpg'))
    st.image(image, use_column_width=True)

    max_duration = 15  # Adjust for 15 seconds

    if st.button('Record'):
        with st.spinner(f'Recording for {max_duration} seconds ....'):
            record(max_duration)  # Pass max_duration to record function
        st.success("Recording completed")

    uploaded_file = st.file_uploader("Upload a WAV file", type="wav")

    if uploaded_file is not None:
        with open(WAVE_OUTPUT_FILE, "wb") as f:
            f.write(uploaded_file.getvalue())

    if st.button('Play'):
        try:
            audio_file = open(WAVE_OUTPUT_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio/wav')  # Play the entire audio file
        except FileNotFoundError:
            st.write("Please record sound first")

    if st.button('Classify'):
        if not os.path.exists(WAVE_OUTPUT_FILE):
            st.write("Please record sound first")
            return

        cnn = init_model()
        with st.spinner("Classifying the chord"):
            ps, _ = get_spectrogram(WAVE_OUTPUT_FILE)
            ps = np.expand_dims(ps, axis=-1)  # Add a channel dimension
            chord_probabilities = cnn.predict(ps)
            chord_index = np.argmax(chord_probabilities)
            chord = CLASSES[chord_index] if chord_index < len(CLASSES) else 'N/A'

        st.success("Classification completed")

        if chord == 'N/A':
            st.write("No chord detected or issue with recording")
        else:
            st.write("### The recorded chord is **", chord + "**")
        st.write("\n")

    # Optional spectrogram display
    if st.button('Display Spectrogram'):
        if os.path.exists(WAVE_OUTPUT_FILE):
            spectrogram, format = get_spectrogram(WAVE_OUTPUT_FILE)
            display(spectrogram, format)
        else:
            st.write("Please record sound first")


if __name__ == '__main__':
    main()
