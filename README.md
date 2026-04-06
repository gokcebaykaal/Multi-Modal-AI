# 🧠 Multi-Modal AI

## 📌 Proje Tanımı

Bu proje, kullanıcıdan alınan **görsel, PDF ve metin girdilerini işleyerek anlamlı ve açıklanabilir çıktılar üreten multi-modal bir yapay zekâ sistemidir**.

Sistem aşağıdaki üç ana yapay zekâ alanını birleştirir:

- 🖼️ Computer Vision (Görüntü İşleme)
- 📄 Retrieval-Augmented Generation (RAG)
- 🧠 Explainable AI (Grad-CAM)

Amaç, farklı veri türlerini tek bir sistem altında işleyebilen, kullanıcıya **tek ve birleşik cevap sunabilen ölçeklenebilir bir AI platformu** geliştirmektir.

---

## 🎯 Proje Amaçları

- Multi-modal veri işleme (image + pdf + text)
- Explainable AI (Grad-CAM)
- Retrieval tabanlı doküman analizi (RAG)
- FastAPI ile ölçeklenebilir backend
- Tek endpoint üzerinden birleşik sistem davranışı

---

# 🏗️ Sistem Mimarisi
Frontend (HTML + JS)
↓
FastAPI Backend
↓
| Image Service (CV) |
| Document Service (RAG) |
↓
Model + FAISS + GradCAM

---

## 🧩 Bileşenler

### 1️⃣ Computer Vision Modülü

- Model: MobileNetV2 (Transfer Learning)
- Çıktılar:
  - Sınıflandırma sonucu
  - Confidence score
  - Grad-CAM görsel açıklama

---

### 2️⃣ RAG Modülü (PDF & Metin)

#### Pipeline:

1. PDF → Text Extraction (PyMuPDF)
2. Chunking
3. Embedding (Sentence Transformers)
4. Similarity Search (FAISS)
5. Answer Generation

---

### 3️⃣ Multi-Modal Router

- Girdi türünü otomatik algılar:
  - image → CV
  - pdf → RAG
  - text → NLP
- Tek endpoint: `/multi-query`

---

# ⚙️ Kullanılan Teknolojiler

| Teknoloji | Açıklama |
|----------|--------|
| FastAPI | Backend framework |
| PyTorch | Deep Learning |
| Torchvision | Model |
| Sentence Transformers | Embedding |
| FAISS | Vector Search |
| PyMuPDF | PDF parsing |
| OpenCV | Grad-CAM |
| HTML/CSS/JS | Frontend |

---

# 🚀 Kurulum Adımları

## 1. Repository klonla

```bash
git clone https://github.com/kullaniciadi/multi-modal-ai.git
cd multi-modal-ai/backend 
```

## 2. Sanal ortam oluştur

python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

## 3. Gerekli paketleri yükle

pip install -r requirements.txt

## 4. Backend başlat

uvicorn app.main:app --reload

## 5. Frontend çalıştır

frontend/index.html dosyasını tarayıcıda aç

## 6. Swagger UI

http://127.0.0.1:8000/docs

---

### 📡 API Dokümantasyonu

## 🔹 1. Görsel Analiz

Endpoint:
POST /analyze-image

Input:
image (file)

Output:
{
  "label": "dog",
  "confidence": 92.3,
  "gradcam": "base64..."
}

## 🔹 2. PDF Soru-Cevap

Endpoint:
POST /ask-document

Input:
file: PDF
question: string

Output:
{
  "question": "...",
  "answer": "...",
  "sources": [...]
}

## 🔹 3. Multi-Modal Endpoint

Endpoint:
POST /multi-query

Input:
image / pdf / text (kombinasyon)

Output:
{
  "mode": "image+text",
  "answer": "...",
  "label": "...",
  "confidence": 90
}

---

### 🧪 Kullanım Senaryoları

## 🖼️ Görsel Analiz
-Kullanıcı görsel yükler
-Sistem sınıflandırır
-Grad-CAM ile açıklama sunar

## 📄 PDF Analizi
-Kullanıcı PDF yükler
-Soru sorar
-Sistem ilgili parçaları bulur
-Kaynaklı cevap üretir

## 🔀 Multi-Modal Kullanım
-Görsel + soru
-PDF + soru
-Sadece metin

Sistem otomatik yönlendirme yapar ve tek cevap üretir.

---

### 📊 RAG Detayları

Kullanılan yapı:
Text → Chunk → Embedding → FAISS → Retrieval → Answer

Not:
FAISS in-memory çalışır
Kalıcı vector DB yoktur
LLM entegrasyonu opsiyoneldir

---

## 📦 Örnek Veri

Test için:
herhangi bir PDF
ImageNet sınıfı içeren görseller yüklenebilir.

---

### 🛠️ Geliştirme Alanları

Redis cache
Celery background jobs
LLM integration
Docker deployment
ChromaDB / Pinecone

---

### 👩‍💻Geliştirici
Gökçe Baykal

