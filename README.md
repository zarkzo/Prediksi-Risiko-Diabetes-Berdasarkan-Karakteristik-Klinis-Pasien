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

Tahapan **data preparation** dilakukan untuk meningkatkan kualitas data sebelum digunakan dalam pemodelan. Langkah-langkah ini bertujuan agar model machine learning yang dibangun dapat belajar dari data secara optimal dan tidak bias akibat data ekstrem, distribusi tidak seimbang, atau perbedaan skala antar fitur.

---

### 1. Pembersihan Data

- **Cek Nilai Kosong dan Duplikat**  
  Dataset dicek untuk mengetahui apakah terdapat nilai kosong (`null`) atau data duplikat. Hasil pengecekan menunjukkan bahwa tidak ada nilai yang hilang dan tidak ditemukan data duplikat.

---

### 2. Deteksi dan Penghapusan Outlier (Z-Score)

- **Metode**:  
  Outlier dideteksi menggunakan **Z-score**, yaitu cara mengukur sejauh mana suatu nilai menyimpang dari rata-rata dalam satuan standar deviasi.  
  Rumus Z-score:  
  $Z = \frac{X - \mu}{\sigma}$

  Di mana:
  - \(X\): nilai fitur,
  - \(\mu\): rata-rata fitur,
  - \(\sigma\): standar deviasi fitur.


- **Alasan Penggunaan Z-Score**:  
  Z-score efektif untuk data yang terdistribusi mendekati normal dan dapat mengidentifikasi outlier secara sistematis.

- **Threshold**:  
  Ambang batas yang digunakan adalah **Z > 3** atau **Z < -3**, artinya data yang menyimpang lebih dari 3 standar deviasi dari rata-rata dianggap sebagai outlier.

- **Tujuan Penghapusan Outlier**:  
  Menghapus data yang sangat ekstrem untuk **mengurangi noise** dan **mencegah model menjadi bias** terhadap nilai-nilai yang tidak representatif terhadap populasi data.

---

### 3. Menyeimbangkan Kelas Target (SMOTE)

- **Masalah**:  
  Dataset awal memiliki distribusi target yang tidak seimbang, yaitu jumlah pasien tanpa diabetes jauh lebih banyak dibandingkan yang menderita diabetes.

- **Solusi**:  
  Menggunakan **SMOTE (Synthetic Minority Over-sampling Technique)** untuk **menyeimbangkan jumlah kelas** dengan menambahkan data sintetis pada kelas minoritas.

- **Tujuan**:  
  Meningkatkan kemampuan model dalam mengenali kelas minoritas (positif diabetes) dan **mengurangi bias terhadap kelas mayoritas**.

---

### 4. Normalisasi Fitur (MinMaxScaler)

- **Alasan**:  
  Beberapa algoritma seperti **KNN dan SVM** sensitif terhadap skala data, sehingga diperlukan normalisasi agar semua fitur berada dalam rentang yang sama.

- **Metode**:  
  Digunakan **Min-Max Scaling** untuk mengubah nilai fitur ke dalam rentang **0 hingga 1**.

  Rumus Min-Max Scaling:  
  $X_{\text{scaled}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$

- **Tujuan**:  
  Menyamakan skala antar fitur agar **jarak antar titik data tidak didominasi oleh fitur dengan skala besar**.

---

### 5. Seleksi Fitur (Recursive Feature Elimination - RFE)

- **Metode**:  
  Digunakan **RFE (Recursive Feature Elimination)** dengan model **Logistic Regression** sebagai estimator untuk memilih **5 fitur terbaik** yang paling berkontribusi terhadap target.

- **Tujuan**:  
  Mengurangi kompleksitas model, meningkatkan performa, dan menghindari overfitting dengan menghilangkan fitur yang kurang relevan.

---

### Kesimpulan Tahapan

Setiap tahapan pada data preparation memiliki peran penting:
- **Membersihkan data** dari noise dan duplikasi.
- **Mengatasi outlier** agar distribusi data tetap konsisten.
- **Menyeimbangkan kelas target** untuk menghindari bias model.
- **Normalisasi** agar algoritma berbasis jarak bekerja optimal.
- **Seleksi fitur** agar model lebih efisien dan akurat.

Seluruh tahapan dilakukan **berurutan dan konsisten dengan notebook**, sehingga data siap digunakan dalam proses modeling secara optimal.

---

## Modeling

### Algoritma

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Komparasi Model

| Algoritma | Mekanisme | Hyperparameter Default Penting | Kelebihan | Kekurangan |
|----------|-----------|------------------------------|-----------|------------|
| **Logistic Regression** | Model klasifikasi linier yang memprediksi probabilitas kelas menggunakan fungsi logistik (sigmoid). Cocok untuk memodelkan hubungan linier antara fitur dan target. | `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, `max_iter=100` | Cepat, sederhana, mudah diinterpretasi | Kurang efektif untuk data non-linear |
| **Random Forest** | Ensemble dari banyak decision tree. Hasil prediksi ditentukan berdasarkan mayoritas voting dari pohon-pohon tersebut. | `n_estimators=100`, `criterion='gini'`, `max_depth=None`, `random_state=None` | Tahan terhadap overfitting, kuat untuk data kompleks | Interpretasi lebih rumit, waktu pelatihan lebih lama |
| **SVM (Support Vector Machine)** | Mencari hyperplane optimal yang memisahkan kelas dengan margin maksimum. Menggunakan kernel untuk menangani data non-linear. | `kernel='rbf'`, `C=1.0`, `gamma='scale'` | Efektif untuk data high-dimensional, performa tinggi pada klasifikasi | Tidak efisien pada data besar, sensitif terhadap scaling dan parameter |
| **K-Nearest Neighbors (KNN)** | Mengklasifikasikan data berdasarkan mayoritas label dari k tetangga terdekat (berdasarkan jarak Euclidean). | `n_neighbors=5`, `weights='uniform'`, `metric='minkowski'`, `p=2` | Sederhana, mudah dipahami | Boros memori, lambat untuk data besar, sensitif terhadap outlier |

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
