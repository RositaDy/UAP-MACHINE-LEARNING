# 🎭 Emotion Classification — Deep Learning Text Classifier

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-blue?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Python-ML%20Project-green?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/NLP-Transformers-orange?logo=transformers">
</p>

## 🌟 Overview

📌 **Emotion Classification** adalah sistem klasifikasi emosi teks yang memanfaatkan beberapa pendekatan Deep Learning modern, termasuk:

✔ Neural Network berbasis **LSTM**  
✔ Transformer *pretrained* seperti **BERT**  
✔ Versi ringan transformer yaitu **DistilBERT**

Sistem ini juga dilengkapi dengan **user interface berbasis web** menggunakan **Streamlit**, sehingga pengguna dapat langsung menginput teks dan melihat prediksi emosinya secara interaktif.

---

## 🔎 Dataset

📂 Dataset yang digunakan adalah dataset emosi berbasis teks yang tersedia di HuggingFace:

➡️ *(https://huggingface.co/datasets/boltuix/emotions-dataset)*

Dataset ini berisi ribuan kalimat berlabel emosi, seperti:
- `disgust`
- `sadness`
- `anger`
- `fear`
- `love`
- `surprise`
- `happiness`
- `neutral`
- `guilt`
- `confusion`
- `desire`
- `shame`
- `sarcasm`


Setiap entri terdiri dari:
| Kolom | Deskripsi |
|-------|-----------|
| `Sentence` | Kalimat atau teks yang diekspresikan |
| `Label` | Label emosi dari teks tersebut |

📌 Dataset ini digunakan untuk melakukan pelatihan dan pengujian model NLP.

---

## 🔎 Exploratory Data Analysis (EDA)
---

### 📈 Analisis Distribusi Data
EDA dilakukan untuk memahami karakteristik data teks dan distribusi kelas emosi.

| Analisis | Visualisasi |
|--------|------------|
| Distribusi Panjang Kalimat | <img width="642" height="393" alt="image" src="https://github.com/user-attachments/assets/25bf9510-72bf-4715-9daa-ea1c8ad26328" />|
| Distribusi Kelas Emosi | <img width="713" height="439" alt="image" src="https://github.com/user-attachments/assets/639e8578-283e-4dd3-b6fe-5d65d6c71df1" />|

📌 **Insight EDA (Ringkas):**
- Panjang kalimat bervariasi dan didominasi teks pendek
- Distribusi kelas tidak sepenuhnya seimbang
- Perlu padding dan tokenisasi sebelum pemodelan

---

## 🧠 Preprocessing Data

Sebelum dilakukan pelatihan model, data dipersiapkan melalui beberapa tahapan singkat berikut:

1. **Lowercase** — Mengkonversi seluruh teks menjadi huruf kecil.
2. **Cleaning** — Menghapus karakter khusus, tanda baca yang tidak relevan.
3. **Tokenisasi**  
   - Untuk LSTM: menggunakan Tokenizer Keras.
   - Untuk BERT/DistilBERT: menggunakan tokenizer spesifik model transformer.
4. **Padding**  
   - Menyamakan panjang input sequence untuk LSTM.
5. **Encode Label**  
   - Mengubah label emosi menjadi representasi numerik untuk pemrosesan model.

---

## 📊 Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| **LSTM** | RNN | Model berbasis LSTM yang menangkap informasi sekuensial dalam teks. |
| **BERT** | Transformer | Model pretrained transformer yang memahami konteks dua arah. |
| **DistilBERT** | Transformer (Lite) | Versi lebih ringan dari BERT dengan performa kompetitif. |

---

## 📈 Evaluation Summary

| Name Model | Accuracy | Analysis results |
|-------|----------|-------|
| **LSTM** | 0.55 | Bagus untuk baseline, kurang peka konteks panjang |
| **BERT** | 0.66 | Akurasi tertinggi dengan pemahaman konteks solid |
| **DistilBERT** | 0.67 | Trade-off cepat + akurasi mendekati BERT |

> **Note:** Gantilah `XX%` dengan angka hasil evaluasi kamu.

---

## 📌 Results Visualization

Berikut tabel tempat kamu bisa sisipkan gambar hasil evaluasi:

### Confusion Matrix & Learning Curves

| Model | Confusion Matrix | Loss Curve | Accuracy Curve |
|-------|------------------|------------|----------------|
| **LSTM** | <img width="528" height="435" alt="image" src="https://github.com/user-attachments/assets/c8306c54-c3ed-4ac5-97dd-dfe95874cdf6" /> | <img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/05d3f875-15db-4a26-bbc4-7dca24973515" /> | <img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/d3e46cd7-61be-44c4-a392-3cc154dce5e3" />|
| **BERT** | <img width="695" height="584" alt="image" src="https://github.com/user-attachments/assets/ece5605a-b07a-46a4-81c6-9572f9515507" /> | <img width="516" height="351" alt="image" src="https://github.com/user-attachments/assets/c8d7afa7-5498-451d-a341-3db2046a684d" /> | <img width="534" height="351" alt="image" src="https://github.com/user-attachments/assets/b3eef260-2aa7-4921-b00e-b0c21992d007" /> |
| **DistilBERT** | <img width="695" height="584" alt="image" src="https://github.com/user-attachments/assets/eeb26357-1fc2-4163-9824-7edc40066e67" /> | <img width="516" height="374" alt="image" src="https://github.com/user-attachments/assets/b56194c4-af05-460a-b7ca-72de146fab1f" /> | <img width="534" height="374" alt="image" src="https://github.com/user-attachments/assets/0f82c736-ac68-4532-b8d9-cefc069538ea" />|

---

## 🛠 How It Works

### 🧾 Input
Pengguna memasukkan teks bebas melalui form **input** pada tampilan web.

### 📌 Output
Setelah diklik tombol _Predict_, sistem akan menampilkan:
- Prediksi kelas emosi
- Confidence score (%)
- Bar chart probabilitas tiap kelas

---

## 🚀 Local Setup & Running Guide

Langkah-langkah singkat untuk menjalankan aplikasi secara lokal:

### 1️⃣ Clone Repository
```bash
git clone https://github.com/RositaDy/UAP-MACHINE-LEARNING.git
cd UAP-MACHINE-LEARNING
```

2️⃣ Install Dependencies
Gunakan file requirements.txt yang sudah tersedia:
```bash
pip install -r requirements.txt
```
⚠️ Disarankan memakai Python 3.10+

3️⃣ Run the Streamlit App
```bash
streamlit run app.py
```
Setelah itu, buka browser dan kunjungi:
```bash
http://localhost:8501
```

| Feature | appearance | 
|-------|----------|
| 📝 Input Form | ** | 
| 📊 Result View | ** | 
| 🔎 Probabilities | ** |

---
###
## Student Info !!!
👩‍💻Rosita Dwi Yulianti
🎓Teknik Informatika — Universitas Muhammadiyah Malang
📘NIM: 202210370311368


