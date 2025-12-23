#                                                       🎭 Emotion Classification — Text emotion classification using LSTM, BERT, and DistilBERT

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

📌 Dataset ini digunakan untuk melakukan pelatihan dan pengujian model ML.

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
| **LSTM** | Neural Network Base (Non-Pretrained) | Model berbasis LSTM yang menangkap informasi sekuensial dalam teks. |
| **BERT** | Model Pretrained (Transfer Learning) | Model pretrained transformer yang memahami konteks dua arah. |
| **DistilBERT** | Model Pretrained (Transfer Learning) | Versi lebih ringan dari BERT dengan performa kompetitif. |

---

## 📈 Evaluation Summary

| Name Model | Accuracy | Analysis results |
|-------|----------|-------|
| **LSTM** | 0.55 | Model LSTM memiliki akurasi terendah karena tidak menggunakan pretrained embedding (seperti GloVe atau FastText), sehingga representasi kata bersifat terbatas. Tokenisasi yang digunakan bersifat statis dan tidak kontekstual, membuat model sulit membedakan emosi dengan makna yang mirip. Berdasarkan kode implementasi, LSTM hanya mengandalkan padding dan embedding dasar, sehingga kurang mampu menangkap hubungan semantik kompleks dalam teks emosi. |
| **BERT** | 0.66 | BERT menunjukkan peningkatan akurasi yang signifikan karena menggunakan pretrained language model dengan mekanisme self-attention dua arah. Hal ini membuat BERT lebih kuat dalam memahami konteks kalimat dan emosi implisit. Namun, performanya masih terbatas akibat fine-tuning yang dilakukan dengan jumlah epoch terbatas dan tanpa GPU (CPU-only), sehingga potensi model belum dimanfaatkan secara maksimal. |
| **DistilBERT** | 0.68 | DistilBERT mampu melampaui BERT karena arsitekturnya yang lebih ringan dan stabil saat pelatihan terbatas. Model ini menggunakan pendekatan knowledge distillation, sehingga mempertahankan informasi penting dari BERT dengan jumlah parameter yang lebih sedikit. Akibatnya, DistilBERT memiliki generalisasi yang lebih baik dan risiko overfitting yang lebih rendah, serta sangat cocok untuk implementasi sistem berbasis Streamlit. |


---

## 📌 Results Visualization

Berikut tabel  hasil evaluasi dari ketiga model:

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
| ℹ️ Sidebar | <img width="498" height="1526" alt="image" src="https://github.com/user-attachments/assets/af92635e-17ce-4f19-bdea-afa299d08933" /> |
| 📝 Input Form | <img width="2880" height="1563" alt="image" src="https://github.com/user-attachments/assets/5dbde0ea-e4c9-41d8-ba3b-9ef9650d9747" /> | 
| 📊 Result View | <img width="1174" height="628" alt="image" src="https://github.com/user-attachments/assets/437b0f20-cedc-4f3e-b0b6-3ee93ade7ab1" /> | 
| 🔎 Probabilities |<img width="1154" height="828" alt="image" src="https://github.com/user-attachments/assets/a6df1e8b-b078-447f-bbed-257f4e27b721" /> |

---
###
## Link Live Demo:
[Streamlit App][(Link Dashboard)](https://uap-machine-learning-w2nrkr37xtxaq2shkxqpue.streamlit.app/)

---
###
## 📚 Student Info !!!
👩‍💻Rosita Dwi Yulianti
🎓Teknik Informatika — Universitas Muhammadiyah Malang
📘NIM: 202210370311368


