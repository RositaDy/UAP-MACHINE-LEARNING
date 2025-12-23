# =====================================================
# Emotion Classification App (FINAL - PYTHON 3.12 SAFE)
# =====================================================

import streamlit as st
import numpy as np
import pickle
import torch
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
# =====================================================
# KONFIGURASI
# =====================================================
MAX_LEN_LSTM = 100
DEVICE = "cpu"
torch.set_num_threads(1)

st.set_page_config(
    page_title="Emotion Classification",
    page_icon="🎭",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.model_name = None

# =====================================================
# CSS (KONTRAS TINGGI)
# =====================================================
st.markdown("""
<style>
.stApp { background-color: #ffffff; color: #1f2937; }
.header-box {
    background: linear-gradient(90deg, #1e3a8a, #2563eb);
    padding: 26px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}
textarea {
    background-color: #f9fafb !important;
    color: #111827 !important;
}
.stButton > button {
    background-color: #2563eb;
    color: white !important;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
}
.stButton > button:hover { background-color: #1e40af; }
.result-card {
    background-color: #f3f4f6;
    padding: 20px;
    border-radius: 16px;
    border-left: 6px solid #2563eb;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="header-box">
    <h1>🎭 Emotion Classification System</h1>
    <p>Klasifikasi emosi teks menggunakan LSTM, BERT, dan DistilBERT</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LOAD LABEL ENCODER
# =====================================================
with open("encoders/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

class_names = label_encoder.classes_

# =====================================================
# MODEL LOADER
# =====================================================
def load_lstm():
    model = tf.keras.models.load_model("Model/LSTM/model_lstm.h5")
    with open("Model/LSTM/tokenizer_lstm.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(class_names)
    )
    model.eval()
    return model, tokenizer

def load_distilbert():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(class_names)
    )
    model.eval()
    return model, tokenizer

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("ℹ️ Tentang Sistem")
st.sidebar.write(
    """
    Sistem ini merupakan aplikasi **klasifikasi emosi teks**
    berbasis **Machine Learning** dan **Deep Learning**.
    
    Model yang digunakan:
    - LSTM (Non-Pretrained)
    - BERT
    - DistilBERT
    """
)

st.sidebar.markdown("---")

st.sidebar.header("⚙️ Pengaturan Model")
model_choice = st.sidebar.selectbox(
    "Pilih Model",
    ["LSTM (Non-Pretrained)", "BERT", "DistilBERT"]
)

st.sidebar.markdown("---")
st.sidebar.caption("👩‍💻 Dikembangkan oleh **Rosita Dwi Yulianti**")

# RESET MODEL JIKA PILIHAN BERUBAH
if st.session_state.model_name != model_choice:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.model_name = model_choice

# =====================================================
# MAIN UI
# =====================================================
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("📝 Input Teks")
    text_input = st.text_area(
        "Masukkan kalimat:",
        height=200,
        placeholder="Contoh: I feel very happy today..."
    )
    predict_btn = st.button("🔍 Prediksi Emosi", use_container_width=True)

with col2:
    st.subheader("📊 Hasil Prediksi")

    if predict_btn:
        if not text_input.strip():
            st.warning("⚠️ Teks tidak boleh kosong.")
            st.stop()

        try:
            with st.spinner("Memproses prediksi..."):

                if st.session_state.model is None:
                    if model_choice == "LSTM (Non-Pretrained)":
                        model, tokenizer = load_lstm()
                    elif model_choice == "BERT":
                        model, tokenizer = load_bert()
                    else:
                        model, tokenizer = load_distilbert()

                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer

                model = st.session_state.model
                tokenizer = st.session_state.tokenizer

                if model_choice == "LSTM (Non-Pretrained)":
                    seq = tokenizer.texts_to_sequences([text_input])
                    padded = pad_sequences(seq, maxlen=MAX_LEN_LSTM)
                    probs = model.predict(padded, verbose=0)[0]
                else:
                    inputs = tokenizer(
                        text_input,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=128
                    )
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

            pred_idx = np.argmax(probs)
            pred_label = class_names[pred_idx]
            confidence = probs[pred_idx] * 100

            st.markdown(
                f"""
                <div class="result-card">
                    <h3>🎯 Prediksi Emosi</h3>
                    <h2 style="color:#1e40af;">{pred_label.upper()}</h2>
                    <p><b>Confidence:</b> {confidence:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 📈 Probabilitas Kelas")
            st.bar_chart({
                class_names[i]: float(probs[i])
                for i in range(len(class_names))
            })

        except Exception as e:
            st.error("Terjadi error saat prediksi.")
            st.code(str(e))
