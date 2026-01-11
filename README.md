# Deep Learning Final Term - Hands-on End-to-End Models

Repository ini berisi kumpulan tugas akhir (UAS) mata kuliah Deep Learning yang fokus pada implementasi end-to-end Natural Language Processing (NLP) menggunakan arsitektur Transformer dari HuggingFace.

## 📋 Overview

Proyek ini mengeksplorasi tiga paradigma utama dalam NLP modern melalui implementasi praktis:
- **Encoder Models** (BERT family) untuk klasifikasi teks
- **Encoder-Decoder Models** (T5) untuk question answering
- **Decoder-only Models** (Phi-2) untuk text summarization

Setiap task diimplementasikan sebagai repository terpisah dengan pipeline lengkap mulai dari preprocessing data, fine-tuning model, evaluasi, hingga analisis performa.

## 🎯 Tasks Repository

### Task 1: Text Classification with BERT
**Repository**: [fine-tuning-bert-text-classification](link-ke-repo-task1)

Fine-tuning BERT-family models untuk klasifikasi teks menggunakan dataset AG News, GoEmotions, dan MultiNLI.

**Model Options**: BERT / DistilBERT / TinyBERT  
**NLP Problem**: Text Classification (NLI)

---

### Task 2: Question Answering with T5
**Repository**: [t5-question-answering](link-ke-repo-task2)

Implementasi generative question answering menggunakan T5-base dengan pendekatan sequence-to-sequence modeling pada dataset SQuAD.

**Model**: T5-base  
**NLP Problem**: Question Answering

---

### Task 3: Text Summarization with Phi-2
**Repository**: [phi2-text-summarization](link-ke-repo-task3)

Fine-tuning decoder-only LLM (Phi-2) untuk menghasilkan ringkasan abstraktif menggunakan dataset XSum.

**Model**: Phi-2  
**NLP Problem**: Abstractive Text Summarization

## 👥 Team Information

**Kelompok**: [Nomor Kelompok]  
**Anggota**:
- Muhamad Mario Rizki - 1103223063 - TK-46-02 - Task 1
- Raihan Ivando Diaz - 1103223093 - TK-46-02 - Task 2
- Abid Sabyano Rozhan - 1103220222 - TK-46-02 - Task 3

## 📚 Learning Objectives

Proyek ini bertujuan untuk meningkatkan pemahaman praktis dalam:
- Implementasi end-to-end deep learning pipeline
- Fine-tuning pre-trained Transformer models dari HuggingFace
- Data preprocessing dan tokenization untuk NLP
- Model evaluation dan performance comparison
- Perbandingan arsitektur: encoder vs encoder-decoder vs decoder-only

## 🛠️ Technology Stack

- **Framework**: PyTorch / TensorFlow
- **Model Hub**: HuggingFace Transformers
- **Environment**: Python, Jupyter Notebook
- **Libraries**: transformers, datasets, tokenizers, evaluate

## 📊 Repository Structure

Setiap task repository mengikuti struktur standar:
```bash
task-repository/
├── notebooks/ # Jupyter notebooks dengan penjelasan lengkap
├── data/ # Dataset dan preprocessing scripts
├── models/ # Trained models dan checkpoints
├── results/ # Evaluation metrics dan analysis
└── README.md # Dokumentasi task-specific
```
## 🚀 Getting Started

1. Clone repository task yang ingin Anda jalankan
2. Install dependencies: `pip install -r requirements.txt`
3. Buka Jupyter notebook untuk melihat implementasi lengkap
4. Follow step-by-step instructions di dalam notebook

## 📝 Submission Info

**Deadline**: 12 Januari 2025
**Status**: ✅ Completed

**Note**: Setiap repository task memiliki README.md detail dengan penjelasan spesifik tentang implementasi, hasil evaluasi, dan cara menjalankan kode [file:4].
