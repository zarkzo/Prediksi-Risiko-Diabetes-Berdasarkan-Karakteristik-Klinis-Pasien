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

Tahapan **data preparation** yang dilakukan adalah sebagai berikut:

1. **Deteksi dan Penanganan Outlier**:

   - Menggunakan **IQR (Interquartile Range)** untuk mendeteksi dan menghapus outlier dari dataset agar model tidak terpengaruh oleh data ekstrem.

2. **SMOTE**:

   - Dataset yang tidak seimbang (jumlah pasien dengan diabetes lebih sedikit daripada yang tidak) ditangani dengan **SMOTE (Synthetic Minority Over-sampling Technique)**, yang menambah sampel sintetis untuk kelas minoritas (diabetes).

3. **Normalisasi**:

   - Data dinormalisasi menggunakan **Min-Max Scaling** agar semua fitur berada dalam rentang yang sama (0 hingga 1), yang penting untuk algoritma berbasis jarak seperti **KNN** dan **SVM**.

4. **Feature Selection**:
   - Fitur yang tidak relevan dihapus dan fitur yang penting dipilih menggunakan **Recursive Feature Elimination (RFE)** untuk meningkatkan kinerja model.

---

## Modeling

Tahapan **pemodelan** yang dilakukan adalah:

1. **Modeling dengan Logistic Regression, Random Forest, dan SVM**:
   - Masing-masing model diterapkan untuk memprediksi risiko diabetes berdasarkan fitur yang ada. Kelebihan **Random Forest** adalah kemampuannya menangani dataset besar dan tidak terstruktur, sementara **SVM** efektif dalam klasifikasi data non-linear, dan **Logistic Regression** memberikan interpretasi yang mudah.
2. **Hyperparameter Tuning**:
   - **Random Forest** dan **SVM** di-tune menggunakan **GridSearchCV** untuk menemukan parameter terbaik (misalnya, jumlah pohon dalam Random Forest atau parameter kernel dalam SVM).

---

## Evaluation

### Metrik Evaluasi

Beberapa metrik evaluasi yang digunakan untuk mengukur kinerja model adalah:

- **Accuracy**: Persentase prediksi yang benar dari total prediksi.
- **Precision**: Persentase prediksi yang benar untuk kelas positif (diabetes).
- **Recall**: Kemampuan model untuk menemukan semua kasus positif (diabetes).
- **F1-Score**: Harmonik rata-rata antara **precision** dan **recall**, sangat berguna ketika dataset tidak seimbang.

### Hasil Evaluasi

- **Model yang terbaik** berdasarkan **akurasi** dan **F1-score** adalah **Random Forest**, diikuti oleh **Logistic Regression** dan **SVM**.
- **Confusion Matrix** memberikan gambaran mengenai seberapa baik model mengklasifikasikan **diabetes** dan **tidak diabetes**.

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
