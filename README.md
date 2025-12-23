🎭 Emotion Classification using LSTM, BERT, and DistilBERT
<p align="center"> <b>Deep Learning–based Emotion Classification System with Web Deployment</b><br> <i>Machine Learning Final Project</i> </p>
📑 Table of Contents

Deskripsi Proyek

Dataset dan Preprocessing

Exploratory Data Analysis (EDA)

Model yang Digunakan

Hasil Evaluasi dan Analisis Perbandingan

Visualisasi Hasil Model

Panduan Menjalankan Sistem Secara Lokal

Link Live Demo (Optional)

📚 Deskripsi Proyek

Proyek ini bertujuan untuk membangun sistem klasifikasi emosi berbasis teks menggunakan pendekatan Deep Learning. Sistem mampu mengklasifikasikan emosi dari sebuah teks input ke dalam beberapa kelas emosi tertentu dengan membandingkan performa tiga model utama, yaitu:

LSTM (Long Short-Term Memory)

BERT (Bidirectional Encoder Representations from Transformers)

DistilBERT

Selain proses pelatihan dan evaluasi model, proyek ini juga mengimplementasikan aplikasi web berbasis Streamlit sehingga pengguna dapat melakukan prediksi emosi secara interaktif.

📊 Dataset dan Preprocessing
🔹 Dataset

Dataset yang digunakan berupa dataset klasifikasi emosi berbasis teks, yang berisi pasangan data:

Teks kalimat

Label emosi

Dataset disimpan dalam file:

emotions_dataset.csv

🔹 Preprocessing Data

Tahapan preprocessing yang diterapkan meliputi:

Case folding (lowercasing)

Penghapusan karakter khusus dan simbol

Tokenisasi teks

Padding dan truncation (khusus BERT & DistilBERT)

Label encoding untuk kelas emosi

Pembagian data menjadi data latih dan data uji

Untuk model LSTM, preprocessing tambahan berupa:

Tokenization berbasis word index

Sequence padding

🔍 Exploratory Data Analysis (EDA)

Tahapan EDA dilakukan untuk memahami karakteristik data sebelum pemodelan.

📌 Ringkasan EDA
Analisis	Visualisasi
Distribusi Panjang Kalimat	(Gambar distribusi panjang kalimat teks)
Distribusi Kelas Emosi	(Gambar distribusi kelas emosi)

📌 Catatan: Gambar akan disisipkan secara manual pada bagian ini.

🤖 Model yang Digunakan
1️⃣ LSTM (Long Short-Term Memory)

Model deep learning berbasis RNN

Dilatih dari awal menggunakan dataset emosi

Cocok untuk menangkap dependensi sekuensial pada teks

Model disimpan dalam format .h5

2️⃣ BERT

Pretrained transformer model

Fine-tuning dilakukan untuk evaluasi

Pada deployment, model di-load langsung dari HuggingFace pretrained model

Digunakan untuk memahami konteks dua arah dalam teks

3️⃣ DistilBERT

Versi ringan dan lebih efisien dari BERT

Memiliki performa yang kompetitif dengan ukuran model lebih kecil

Lebih cepat saat inferensi

📈 Hasil Evaluasi dan Analisis Perbandingan
🔹 Tabel Analisis Perbandingan Model (WAJIB)
Nama Model	Akurasi	Hasil Analisis
LSTM	XX%	Performa stabil, namun kurang optimal dalam memahami konteks panjang
BERT	XX%	Memberikan performa terbaik dengan pemahaman konteks yang sangat baik
DistilBERT	XX%	Performa mendekati BERT dengan efisiensi komputasi lebih tinggi

📌 Nilai akurasi diisi berdasarkan hasil eksperimen masing-masing model.

🧪 Visualisasi Hasil Model
📊 Confusion Matrix, Loss, dan Accuracy
Model	Confusion Matrix	Plot Loss	Plot Accuracy
LSTM	(Gambar Confusion Matrix LSTM)	(Plot Loss LSTM)	(Plot Accuracy LSTM)
BERT	(Gambar Confusion Matrix BERT)	(Plot Loss BERT)	(Plot Accuracy BERT)
DistilBERT	(Gambar Confusion Matrix DistilBERT)	(Plot Loss DistilBERT)	(Plot Accuracy DistilBERT)

📌 Catatan: Gambar visualisasi akan ditambahkan secara manual pada tabel ini.

🚀 Panduan Menjalankan Sistem Secara Lokal
🔧 1. Clone Repository
git clone https://github.com/RositaDy/UAP-MACHINE-LEARNING.git
cd UAP-MACHINE-LEARNING

🔧 2. Instalasi Dependensi

Pastikan Python 3.10+ telah terinstal.

pip install -r requirements.txt

🔧 3. Menjalankan Aplikasi Streamlit
streamlit run app.py


Aplikasi akan berjalan secara lokal dan dapat diakses melalui browser.

🌐 Link Live Demo (Optional)

🔗 Live Demo: (akan ditambahkan jika sudah tersedia)

👤 Biodata

Nama: Rosita Dwi Yulianti
NIM: 202210370311368
Program Studi: Teknik Informatika
Universitas: Universitas Muhammadiyah Malang

⭐ Penutup

Repository ini dikembangkan sebagai bagian dari tugas akhir mata kuliah Machine Learning, dengan fokus pada perbandingan performa model deep learning dalam tugas klasifikasi emosi berbasis teks serta implementasi sistem berbasis web.
