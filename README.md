# Medical NER: DistilBERT vs BERT
**Real-time Clinical Entity Recognition System for Mobile Deployment**

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](MASUKKAN_LINK_HUGGING_FACE_KAMU_DISINI)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

## Project Overview
Project ini bertujuan untuk mengembangkan sistem **Named Entity Recognition (NER)** medis yang ringan namun akurat. Fokus utama penelitian ini adalah membandingkan performa model **BERT (Heavy)** dengan **DistilBERT (Light)** untuk mengekstraksi informasi entitas klinis dari teks rekam medis yang tidak terstruktur.

Sistem ini dirancang untuk mengenali tiga entitas utama:
1.  🟦 **Chemical:** Obat-obatan dan zat kimia.
2.  🟥 **Disease:** Penyakit dan gejala klinis.
3.  🟩 **Dosage:** Dosis dan aturan pakai (Menggunakan pendekatan Hybrid Regex).

## Key Results (The "Why")
Mengapa memilih DistilBERT? Eksperimen kami membuktikan bahwa penurunan ukuran model sebesar 40% hanya mengurangi akurasi sebesar 1%, namun memberikan peningkatan kecepatan inferensi hingga **2x lipat**.

| Metric |  BERT (Benchmark) |  DistilBERT (Selected) |  Impact |
| :--- | :--- | :--- | :--- |
| **F1-Score (Accuracy)** | 87.0% | **86.0%** | -1% (Acceptable Trade-off) |
| **Inference Latency** | ~9.99 ms | **~5.13 ms** |  **48.6% Faster** |
| **Model Size** | ~420 MB | **~240 MB** |  **42% Smaller** |

> *Kesimpulan: DistilBERT sangat ideal untuk implementasi pada perangkat dengan sumber daya terbatas (Mobile/Edge Devices).*

## Methodology
### 1. Dataset
Menggunakan **BC5CDR (BioCreative V CDR)**, standar emas untuk dataset Chemical & Disease.

### 2. Model Architecture
- **Training:** Fine-tuning `distilbert-base-cased` selama 10 Epochs menggunakan GPU.
- **Optimization:** Model dikonversi ke format ONNX dan dikompresi (Quantization) untuk kompatibilitas Android.

### 3. Hybrid Logic Implementation
Salah satu tantangan model NLP adalah mengenali pola angka presisi (Dosis). Sistem ini menggunakan pendekatan **Hybrid**:
- **AI (Deep Learning):** Menangani konteks abstrak (Nama Obat vs Penyakit).
- **Regex (Rule-Based):** Menangani pola terstruktur (Angka + Satuan, misal: *500mg*, *3x daily*).

## Repository Structure
```bash
Medical-NER-Portfolio/
├── notebooks/                  # Jupyter Notebooks untuk eksperimen
│   ├── 1_Training_BERT_Heavy.ipynb
│   ├── 2_Training_DistilBERT_Light.ipynb
│   └── 3_Demo_Local.ipynb
├── deployment/                 # Source code untuk Hugging Face Space
│   ├── app.py                  # Logika Gradio Web App
│   └── requirements.txt        # Dependencies
├── images/                     # Aset gambar untuk README
│   └── demo_result.png
└── README.md                   # Dokumentasi Proyek
```
## Acknowledgments
- Dataset provided by BioCreative V.
- Model pre-trained by Hugging Face Transformers. (https://huggingface.co/spaces/Jenny0412/Medical-NER-DistilBERT)

 ---
 Created by Ratu Rinjanhei Macinnes - Informatics Student
 

