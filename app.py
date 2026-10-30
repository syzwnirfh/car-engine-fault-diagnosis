
import json
import tempfile
from pathlib import Path

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT / "models"

MODEL_FILES = {
    "Idle": "idle_with_aug_150_patience20_best.keras",
    "Start-up": "startup_with_aug_best.keras",
}

CLASSES = {
    "Idle": [
        "low_oil",
        "normal_engine_idle",
        "power_steering",
        "serpentine_belt",
    ],
    "Start-up": [
        "bad_ignition",
        "dead_battery",
        "normal_engine_startup",
    ],
}

DISPLAY_NAMES = {
    "low_oil": "Low Oil",
    "normal_engine_idle": "Normal Engine Idle",
    "power_steering": "Power Steering",
    "serpentine_belt": "Serpentine Belt",
    "bad_ignition": "Bad Ignition",
    "dead_battery": "Dead Battery",
    "normal_engine_startup": "Normal Engine Start-up",
}

with open(PROJECT / "preprocessing_config.json", "r") as f:
    CONFIG = json.load(f)

st.set_page_config(
    page_title="Car Engine Fault Diagnosis",
    page_icon="🚗",
    layout="centered",
)

@st.cache_resource
def load_model(state):
    path = MODEL_DIR / MODEL_FILES[state]
    return tf.keras.models.load_model(path, compile=False)



def preprocess_audio(audio_path, state):
    sr = CONFIG["sample_rate"]
    duration_key = "idle" if state == "Idle" else "startup"
    duration = CONFIG["duration"][duration_key]
    segment_length = int(sr * duration)

    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    if len(y) == 0:
        raise ValueError("Audio file is empty.")

    segments = []

    for start in range(0, len(y), segment_length):
        segment = y[start:start + segment_length]

        # Skip incomplete final segment
        if len(segment) < segment_length:
            break

        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            n_fft=CONFIG["n_fft"],
            hop_length=CONFIG["hop_length"],
            n_mels=CONFIG["n_mels"],
            fmax=CONFIG["fmax"]
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_min = mel_db.min()
        mel_max = mel_db.max()

        if mel_max > mel_min:
            mel_normalized = (mel_db - mel_min) / (mel_max - mel_min)
        else:
            mel_normalized = np.zeros_like(mel_db)

        mel_uint8 = (mel_normalized * 255).astype(np.uint8)

        img = Image.fromarray(mel_uint8)
        img = img.resize(
            tuple(CONFIG["image_size"]),
            Image.Resampling.BILINEAR
        )

        x = np.asarray(img, dtype=np.float32) / 255.0
        segments.append(x[:, :, np.newaxis])

    if not segments:
        raise ValueError("Audio is shorter than one complete segment.")

    return np.stack(segments), duration

st.title("🚗 Automated Car Engine Fault Diagnosis")
st.write(
    "Audio-Based Car Engine Fault Classification "
    "Using Convolutional Neural Networks"
)

state = st.radio(
    "Select engine operating state:",
    ["Idle", "Start-up"],
    horizontal=True,
)

uploaded_file = st.file_uploader(
    "Upload car engine audio (.wav or .mp3)",
    type=["wav", "mp3"],
)

recorded_file = st.audio_input("Record engine audio")

audio_file = recorded_file if recorded_file is not None else uploaded_file

if audio_file is not None:
    st.audio(audio_file, format="audio/mpeg" if uploaded_file.name.lower().endswith(".mp3") else "audio/wav")

    if st.button("Analyze Engine Audio", type="primary"):
        with st.spinner("Analyzing audio..."):
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(audio_file.name).suffix.lower(), delete=False
                ) as temp_file:
                    temp_file.write(audio_file.getvalue())
                    temp_path = Path(temp_file.name)


                x, segment_duration = preprocess_audio(
                    str(temp_path), state
                )

                model = load_model(state)

                all_probabilities = model.predict(
                    x, batch_size=32, verbose=0
                )

                # Average probability across all segments
                probabilities = np.mean(
                    all_probabilities, axis=0
                )

                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASSES[state][predicted_index]
                confidence = float(probabilities[predicted_index])

                st.subheader("Prediction Result")
                st.success(DISPLAY_NAMES[predicted_class])

                st.metric(
                    "Average Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.write(
                    f"Total segments analyzed: {len(x)}"
                )

                st.subheader("Overall Class Probabilities")

                for label, probability in zip(
                    CLASSES[state], probabilities
                ):
                    st.write(
                        f"{DISPLAY_NAMES[label]}: "
                        f"{probability * 100:.2f}%"
                    )
                    st.progress(float(probability))

                st.subheader("Prediction for Each Segment")

                for i, segment_probs in enumerate(all_probabilities):
                    index = int(np.argmax(segment_probs))
                    label = CLASSES[state][index]
                    score = float(segment_probs[index])

                    start_time = i * segment_duration
                    end_time = (i + 1) * segment_duration

                    st.write(
                        f"{start_time:.1f}s - {end_time:.1f}s: "
                        f"{DISPLAY_NAMES[label]} "
                        f"({score * 100:.2f}%)"
                    )

                st.info(
                    "Results indicate potential fault categories "
                    "based on audio patterns and do not confirm "
                    "a mechanical diagnosis."
                )

            except Exception as e:
                st.error(f"Analysis failed: {e}")

            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
