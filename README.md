# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding  
Perusahaan Edutech “Jaya Jaya Learning” menyediakan platform pembelajaran daring untuk mahasiswa program sarjana di berbagai jurusan. Misi mereka adalah menurunkan angka dropout dan meningkatkan tingkat kelulusan dengan intervensi tepat waktu. Saat ini belum ada sistem pemantauan risiko dan dashboard yang dapat membantu tim akademik mengenali mahasiswa berisiko tinggi sehingga intervensi sering terlambat.

### Permasalahan Bisnis  
1. **Tingginya angka dropout** (~32 % dari total mahasiswa)  
2. **Kurangnya visibilitas**: tidak ada dashboard real‑time untuk memonitor risiko per segmen (course, marital status, beasiswa, dsb.)  
3. **Intervensi terlambat**: tim akademik hanya tahu mahasiswa bermasalah setelah nilai semester turun drastis  
4. **Sumber daya terbatas**: harus memprioritaskan program mentoring dan beasiswa ke mahasiswa yang paling membutuhkan  

### Cakupan Proyek  
- Data preprocessing dan EDA pada dataset “Students’ Performance”  
- Pengembangan model klasifikasi (Logistic Regression, RandomForest, Gradient Boosting) untuk memprediksi risiko dropout  
- Pembuatan dashboard analitik di Metabase untuk business monitoring  
- Pembuatan prototype Streamlit untuk prediksi real‑time oleh pengguna (admin akademik)  
- Export data ke Supabase PostgreSQL sebagai data source terpusat  

## Persiapan  

**Sumber data**:  
- `students_performance.csv` (4424 baris, 37 kolom) berisi demografi, prestasi akademik sem‑1 & sem‑2, dan status akhir (Dropout/Graduate/Enrolled).  
- Link Dataset : https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv

**Setup environment**:  

1. Buat virtual environment
```
conda create -n edutech_ml python=3.11 -y
conda activate edutech_ml
```

2. Install dependencies
```
pip install pandas scikit-learn matplotlib seaborn streamlit joblib sqlalchemy psycopg2-binary
```

3. (Optional) untuk Metabase: tidak perlu install, cukup koneksi ke Supabase
- **Credential Supabase (untuk Metabase)**:
  - Host: `aws-0-ap-southeast-1.pooler.supabase.com`
  - Port: `6543`
  - Database: `postgres`
  - User: `postgres.yvfheuwphfzqfsppwybl`
  - Password: `<YOUR-PASSWORD>`
  - Pool mode: `transaction`
```

- **Email dan password Metabase**:
  - Email: adrianramadhan881@gmail.com
  - Password: root123
```
## Business Dashboard
Kami membangun dashboard Student Risk Insights di Metabase, terhubung ke Supabase Postgres.


Dashboard menampilkan 6 question utama:

1. Dropout Rate per Marital Status

2. Dropout vs Scholarship Holder

3. Avg Admission Grade by Application Mode

4. Dropout Rate by Attendance Time

5. Dropout by Parents’ Education

6. Top 5 Courses by Dropout %

Tim bisnis dapat memfilter per course, cohort year, dan segmentasi demografis untuk insight real‑time.

## Menjalankan Sistem Machine Learning
Prototype prediksi risiko mahasiswa disajikan via Streamlit app..

1. Jalankan server Streamlit:
```
    streamlit run app.py
```
2. Pilih mode Upload CSV atau Manual Entry di sidebar.

3. Lihat prediksi status (Dropout/Enrolled/Graduate) dengan warna label.

Link prototype:

https://adrianramadhan-educational-institutions-submission-app-5jb3uh.streamlit.app/

## Conclusion
- Dengan model ML kami dapat memprediksi mahasiswa berisiko dropout dengan akurasi ~77 %.

- Dashboard Metabase memudahkan monitoring segmentasi risiko dan efektivitas intervensi.

- Rekomendasi action items (peer‑mentor, beasiswa targeted, flexible scheduling, parent engagement) disusun berdasarkan insight dashboard.

- Implementasi end‑to‑end (data → ML → dashboard → intervensi) siap dijalankan untuk menurunkan angka dropout minimal 20 % dalam dua semester ke depan.

## Rekomendasi Action Items

| No | Insight Utama | Action Item | KPI / Target | Pemilik | Waktu Implementasi |
|----|---------------|-------------|--------------|---------|---------------------|
| 1 | Mahasiswa single memiliki dropout tertinggi (≈50%) | • Luncurkan program "Peer-Mentor Group" khusus mahasiswa lajang<br>• Sesi onboarding & ice-breaking intensif di minggu pertama kuliah | Turunkan dropout single menjadi ≤30% dalam 1 semester | Kepala Akademik & Bimbingan Mahasiswa | Semester 1 2025/26 |
| 2 | Non-scholarship holder dominasi dropout (~1.200) | • Tawarkan beasiswa mini/subsidi uang kuliah untuk 200 mahasiswa berisiko (GPA < 12)<br>• Sistem notifikasi otomatis tagihan & reminder bagi mahasiswa debtor | Penurunan dropout non-beasiswa 25% dalam 6 bulan | Biro Keuangan & Beasiswa | Q3 2025 |
| 3 | Course "Animation & Multimedia" dan "Management" masuk Top 5 risiko | • Desain modul remedial untuk kedua course (workshop, tutoring)<br>• Sediakan study-group mingguan & Q&A bersama dosen | Retensi di dua course +20% dalam 2 semester | Ketua Program Studi | Mulai Agustus 2025 |
| 4 | 85% dropout terjadi pada daytime attendance | • Uji coba kelas evening/hybrid untuk mata kuliah utama<br>• Fleksibilitas jadwal lab & tutorial bagi mahasiswa bekerja | Dropout daytime turun dari 85% → 60% dalam 1 tahun | Wakil Dekan Akademik | Semester 2 2025/26 |
| 5 | Orang tua dengan pendidikan rendah ("Basic Education") korelasi tinggi dengan dropout | • Selenggarakan "Parent Engagement Workshop" setiap awal semester<br>• Kirim newsletter tips belajar & cara dukungan anak ke orang tua | Penurunan dropout anak dari grup ini 20% dalam 1 tahun | Tim Kemitraan Orang Tua | Q3 2025 |
| 6 | Mahasiswa dengan total_units_approved rendah (<6 units) berisiko | • Monitoring real-time via dashboard Metabase (alert jika units_approved <6)<br>• Bimbingan akademik intensif bagi yang ter-alert | Rata-rata units_approved naik menjadi ≥10 | Tim Academic Advising | Mulai Juni 2025 |
