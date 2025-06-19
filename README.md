# Proyek Prediksi Risiko Diabetes Berdasarkan Karakteristik Klinis Pasien - Putra Faaris Prayoga

## Domain Proyek

Masalah **prediksi risiko diabetes** menjadi sangat penting dalam konteks dunia medis, khususnya untuk mengidentifikasi pasien yang berisiko tinggi mengidap diabetes. Diabetes adalah salah satu penyakit yang paling banyak diderita di seluruh dunia, dan prediksi dini dapat membantu mencegah komplikasi serius seperti kerusakan organ dan gangguan metabolik.

### Mengapa Masalah Ini Harus Diselesaikan:

Prediksi risiko diabetes menggunakan **data medis** yang ada, seperti **tingkat glukosa darah**, **tekanan darah**, **indeks massa tubuh (BMI)**, dan **riwayat keluarga diabetes**, akan sangat membantu dalam meminimalkan risiko dan meningkatkan kualitas hidup pasien. Dengan menggunakan model machine learning, kita dapat mengidentifikasi pasien yang berisiko tinggi diabetes lebih awal dan memberi mereka pengobatan atau perubahan gaya hidup yang tepat.

**Referensi Terkait**:

- [National Diabetes Statistics Report, 2020](https://www.cdc.gov/diabetes/pdfs/data/statistics/national-diabetes-statistics-report.pdf)
- [Deep Learning for Predictive Health Analytics](https://pubmed.ncbi.nlm.nih.gov/28550860/)

---

## Business Understanding

### Problem Statements

Pernyataan masalah utama adalah **bagaimana memprediksi risiko diabetes pada pasien berdasarkan data medis yang ada?** Beberapa pertanyaan utama yang harus dijawab meliputi:

1. **Bagaimana faktor-faktor seperti usia, kadar gula darah, BMI, dan riwayat keluarga mempengaruhi risiko diabetes?**
2. **Apakah ada hubungan signifikan antara fitur-fitur ini dan kondisi diabetes pada pasien?**
3. **Bagaimana kita dapat membangun model yang mampu memprediksi kemungkinan diabetes secara akurat?**

### Goals

Tujuan utama dari proyek ini adalah:

1. Mengembangkan model prediksi yang dapat menentukan apakah seorang pasien berisiko terkena diabetes berdasarkan data medis yang tersedia.
2. Meningkatkan **akurasi** prediksi dengan menggunakan model **machine learning** yang efektif, seperti **Logistic Regression**, **Random Forest**, dan **SVM**.
3. Mengidentifikasi faktor-faktor yang paling berpengaruh dalam prediksi risiko diabetes.

### Solution Statement

Untuk mencapai tujuan tersebut, solusi yang diusulkan adalah sebagai berikut:

1. **Modeling dengan Berbagai Algoritma**:

   - Menggunakan **Logistic Regression**, **Random Forest**, dan **SVM** untuk mengevaluasi model dengan berbagai teknik klasifikasi.
   - Melakukan **hyperparameter tuning** untuk meningkatkan performa model dengan **GridSearchCV** atau **RandomizedSearchCV**.

2. **Evaluasi Model**:
   - Menggunakan **akurasi**, **precision**, **recall**, dan **F1-score** untuk mengevaluasi kinerja model. Semua model akan diuji dengan data yang sudah diseimbangkan menggunakan **SMOTE** dan data yang telah **dinormalisasi**.

---

## Data Understanding

Dataset yang digunakan adalah **Pima Indians Diabetes Dataset**, yang berisi data medis pasien yang digunakan untuk memprediksi apakah pasien tersebut mengidap diabetes.

- **Link Dataset**: [UCI Pima Indians Diabetes Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

### Informasi Dataset:

- **Jumlah Data**: Terdapat **768 entri** dalam dataset, yang masing-masing mewakili data pasien.
- **Jumlah Kolom**: Dataset terdiri dari **9 kolom** yang mencakup fitur-fitur medis pasien dan target (Outcome).

Berikut adalah ringkasan informasi mengenai kolom dan tipe data:

| Kolom                    | Jumlah Data Non-Null | Tipe Data |
| ------------------------ | -------------------- | --------- |
| Pregnancies              | 768                  | int64     |
| Glucose                  | 768                  | int64     |
| BloodPressure            | 768                  | int64     |
| SkinThickness            | 768                  | int64     |
| Insulin                  | 768                  | int64     |
| BMI                      | 768                  | float64   |
| DiabetesPedigreeFunction | 768                  | float64   |
| Age                      | 768                  | int64     |
| Outcome                  | 768                  | int64     |

- **Tipe Data**: Sebagian besar kolom menggunakan **`int64`** kecuali **BMI** dan **DiabetesPedigreeFunction** yang menggunakan **`float64`**.
- **Tidak Ada Nilai Null**: Setiap kolom dalam dataset ini memiliki **768 nilai non-null**, artinya tidak ada data yang hilang dalam dataset ini.

### Variabel-variabel pada dataset ini:

- **Pregnancies**: Jumlah kehamilan yang dialami oleh pasien.
- **Glucose**: Kadar glukosa darah.
- **BloodPressure**: Tekanan darah.
- **SkinThickness**: Ketebalan kulit (sering digunakan untuk mengukur kadar lemak tubuh).
- **Insulin**: Kadar insulin dalam darah.
- **BMI**: Indeks massa tubuh.
- **DiabetesPedigreeFunction**: Riwayat keluarga dengan diabetes.
- **Age**: Usia pasien.
- **Outcome**: Target (0: Tidak diabetes, 1: Diabetes).

### Visualisasi Data

**Exploratory Data Analysis (EDA)** digunakan untuk menggambarkan distribusi data dan hubungan antar fitur, serta untuk memeriksa apakah ada **outlier** atau **nilai yang hilang** dalam dataset. Di sini, kita melakukan **histogram** untuk setiap fitur, serta **heatmap** untuk melihat korelasi antar fitur.

---

## Data Preparation

### Teknik dan Penjelasan

1. **Pemeriksaan nilai null dan duplikasi**  
   - Hasil: Tidak ditemukan nilai null dan data duplikat.

2. **Deteksi dan penanganan outlier**  
   - Metode: **Z-score** dengan threshold ±3.
   - Tujuan: Menghindari pengaruh data ekstrem terhadap model.

3. **Exploratory Data Analysis (EDA)**  
   - Heatmap korelasi menunjukkan fitur yang paling berhubungan dengan diabetes: Glucose, BMI, dan Age.

     ![image](https://github.com/user-attachments/assets/dc1d41e7-1486-48fa-98a9-03506e675d02)

   terlihat inight dari data heatmap diatas:

      - **Glucose** memiliki korelasi tertinggi dengan `Outcome` (**0.48**), menunjukkan bahwa kadar glukosa darah      merupakan prediktor kuat untuk diabetes.
      - **BMI** dan **Age** juga memiliki korelasi moderat dengan `Outcome` masing-masing sebesar **0.30** dan **0.25**.
      - Beberapa fitur seperti **SkinThickness**, **Insulin**, dan **DiabetesPedigreeFunction** memiliki korelasi lemah terhadap `Outcome` (nilai < 0.2).
      - Fitur **Pregnancies** berkorelasi kuat dengan **Age** (**0.57**), yang wajar karena semakin tua usia, umumnya jumlah kehamilan meningkat.

4. **SMOTE (Synthetic Minority Oversampling Technique)**  
   - Mengatasi ketidakseimbangan kelas pada kolom `Outcome`.

5. **Normalisasi**  
   - Digunakan **MinMaxScaler** untuk menyamakan skala fitur (penting untuk KNN dan SVM).

6. **Feature Selection**  
   - Menggunakan **Recursive Feature Elimination (RFE)** untuk memilih 5 fitur terbaik.

7. **Train-test split**  
   - Data dibagi 80% train dan 20% test.

---

## Modeling

### Algoritma

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Komparasi Model

| Model               | Kelebihan                                                 | Kekurangan                                         |
|--------------------|-----------------------------------------------------------|----------------------------------------------------|
| Logistic Regression| Cepat, sederhana, interpretatif                           | Kurang untuk data non-linear                       |
| Random Forest       | Tahan overfitting, bisa menangani fitur penting           | Interpretasi model lebih sulit                     |
| SVM                 | Efektif pada data high-dimensional, non-linear            | Butuh tuning kernel, lambat untuk dataset besar    |
| KNN                 | Sederhana, tidak membutuhkan pelatihan                    | Lambat untuk dataset besar, sensitif terhadap outlier |

---

## Evaluation

### Metrik yang Digunakan

- **Accuracy**: Rasio prediksi yang benar terhadap total.
- **Precision**: Rasio prediksi positif yang benar.
- **Recall**: Kemampuan model menangkap semua kelas positif.
- **F1-Score**: Harmonik antara precision dan recall.

### Hasil Evaluasi

- **Model terbaik: Random Forest**
  - Akurasi: 0.83
  - Recall untuk diabetes: 0.86
  - Precision: 0.83
  - F1-score: 0.84

### Confusion Matrix

Confusion Matrix digunakan untuk mengevaluasi performa klasifikasi model terbaik (Random Forest) pada data uji. Matriks ini menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas.

![image](https://github.com/user-attachments/assets/b3cdc6c6-6045-4153-a782-7655b4d05f65)

sehingga didapat Interpretasi:
- **True Positive (TP):** 86 pasien diabetes terprediksi benar → model mendeteksi diabetes secara akurat.
- **True Negative (TN):** 67 pasien non-diabetes terprediksi benar.
- **False Positive (FP):** 18 pasien non-diabetes salah diklasifikasikan sebagai diabetes.
- **False Negative (FN):** 14 pasien diabetes tidak terdeteksi (diklasifikasikan sebagai non-diabetes).



---

## Kesimpulan

Berdasarkan hasil eksperimen yang dilakukan, **Random Forest** menunjukkan kinerja yang paling baik dalam memprediksi risiko diabetes dibandingkan dengan model lainnya, meskipun **Logistic Regression** juga memberikan hasil yang cukup baik. Penggunaan **SMOTE** untuk menyeimbangkan data terbukti meningkatkan akurasi model, sementara **normalisasi** dan **feature selection** membantu memperbaiki performa model dengan mengurangi variansi data.

---

### Referensi

- [Pima Indians Diabetes Dataset (UCI Machine Learning Repository)](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- [National Diabetes Statistics Report, 2020](https://www.cdc.gov/diabetes/pdfs/data/statistics/national-diabetes-statistics-report.pdf)
- [Deep Learning for Predictive Health Analytics](https://pubmed.ncbi.nlm.nih.gov/28550860/)

---

Laporan ini diharapkan memberikan gambaran yang jelas tentang **analisis data**, **pemodelan machine learning**, dan **evaluasi** yang dilakukan dalam proyek ini. Semoga informasi yang disampaikan dapat digunakan untuk memahami langkah-langkah dalam memprediksi risiko diabetes dan mengembangkan solusi berbasis machine learning yang efektif.
