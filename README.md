# 🧠 Multi-Modal AI Platform

## 📌 Proje Tanımı

Bu proje, kullanıcıdan alınan **görsel, PDF ve metin girdilerini işleyerek anlamlı, açıklanabilir ve bağlamsal çıktılar üreten multi-modal bir yapay zekâ sistemidir**.

Sistem aşağıdaki üç temel AI yaklaşımını bir araya getirir:

- 🖼️ Computer Vision (Görüntü Analizi)
- 📄 Retrieval-Augmented Generation (RAG)
- 🧠 Explainable AI (Grad-CAM)

Amaç, farklı veri türlerini tek bir sistem altında işleyebilen, kullanıcıya **tek ve birleşik bir cevap sunabilen ölçeklenebilir bir AI platformu** geliştirmektir.

---

## 🎯 Proje Amaçları

- Multi-modal veri işleme (image + pdf + text)
- Explainable AI (Grad-CAM)
- Retrieval tabanlı doküman analizi (RAG)
- LLM tabanlı cevap üretimi (opsiyonel)
- FastAPI ile ölçeklenebilir backend mimarisi
- Tek endpoint üzerinden birleşik sistem davranışı

---

# 🏗️ Sistem Mimarisi

| Katman | Açıklama |
|--------|--------|
| Frontend | HTML + CSS + JS |
| Backend | FastAPI |
| AI Katmanı | CV + RAG + NLP |
| Model | MobileNetV2 + Sentence Transformers |
| Vektör DB | FAISS (in-memory) |
| Cache | Redis (opsiyonel) |
| Background Jobs | FastAPI BackgroundTasks |

---

## 🧩 Bileşenler

### 1️⃣ Computer Vision Modülü

- Model: MobileNetV2 (Transfer Learning)
- Kütüphaneler: PyTorch, Torchvision

#### Çıktılar:
- Sınıflandırma sonucu (label)
- Confidence score (%)
- Grad-CAM görsel açıklama (Explainable AI)

---

### 2️⃣ RAG Modülü (PDF & Metin)

#### Pipeline:

1. PDF → Text Extraction (PyMuPDF)
2. Chunking
3. Embedding (Sentence Transformers)
4. Similarity Search (FAISS)
5. Context Retrieval
6. Answer Generation (LLM veya extractive)

#### Özellikler:

- En alakalı chunk’lar bulunur
- Kaynaklı cevap üretimi yapılır
- Cache mekanizması ile hız artırılır

---

### 3️⃣ Multi-Modal Router

- Girdi türünü otomatik algılar:
  - image → CV pipeline
  - pdf → RAG pipeline
  - text → NLP / RAG

- Tek endpoint:
```
POST /multi-query
```

---

## ⚙️ Kullanılan Teknolojiler

| Teknoloji | Açıklama |
|----------|--------|
| FastAPI | Backend framework |
| PyTorch | Deep Learning |
| Torchvision | CV modelleri |
| Sentence Transformers | Embedding |
| FAISS | Vector similarity search |
| PyMuPDF | PDF parsing |
| OpenCV | Grad-CAM |
| Redis | Cache (opsiyonel) |
| HTML/CSS/JS | Frontend |

---

# 🚀 Kurulum Adımları

## 1. Repository klonla

```bash
git clone https://github.com/kullaniciadi/multi-modal-ai.git
cd multi-modal-ai/backend
```

## 2. Sanal ortam oluştur

```bash
python -m venv venv
```

### Aktivasyon:

Windows:
```bash
venv\Scripts\activate
```

Linux / Mac:
```bash
source venv/bin/activate
```

---

## 3. Gerekli paketleri yükle

```bash
pip install -r requirements.txt
```

---

## 4. Backend başlat

```bash
uvicorn app.main:app --reload
```

---

## 5. Frontend çalıştır

frontend/index.html dosyasını tarayıcıda aç

---

## 6. Swagger UI

http://127.0.0.1:8000/docs

---

# 📡 API Dokümantasyonu

## 🔹 1. Görsel Analiz

Endpoint:
```
POST /analyze-image
```

Output:
```json
{
  "label": "dog",
  "confidence": 92.3,
  "gradcam": "base64..."
}
```

---

## 🔹 2. PDF Soru-Cevap

Endpoint:
```
POST /ask-document
```

Output:
```json
{
  "question": "...",
  "answer": "...",
  "sources": [...]
}
```

---

## 🔹 3. Multi-Modal Endpoint

Endpoint:
```
POST /multi-query
```

Output:
```json
{
  "mode": "image+text",
  "answer": "...",
  "label": "...",
  "confidence": 90
}
```

---

# 🧪 Kullanım Senaryoları

## 🖼️ Görsel Analiz
- Kullanıcı görsel yükler  
- Sistem sınıflandırır  
- Grad-CAM ile açıklama sunar  

## 📄 PDF Analizi
- Kullanıcı PDF yükler  
- Soru sorar  
- Sistem ilgili parçaları bulur  
- Kaynaklı cevap üretir  

## 🔀 Multi-Modal Kullanım
- Görsel + soru  
- PDF + soru  
- Sadece metin  

---

# 📊 RAG Detayları

Text → Chunk → Embedding → FAISS → Retrieval → Answer

- FAISS in-memory çalışır  
- Kalıcı vector DB yoktur  
- LLM entegrasyonu opsiyoneldir  

---

# ⚡ Performans

- Cache (Redis) ile hızlandırma  
- BackgroundTasks ile async işlem  
- Chunking ile optimizasyon  

---

# 🛠️ Geliştirme Alanları

- Redis cache geliştirme  
- Celery queue sistemi  
- LLM entegrasyonu (GPT / LLaMA)  
- Docker deployment  
- ChromaDB / Pinecone  

---

# 👩‍💻 Geliştirici

**Gökçe Baykal**

---

# ⭐ Not

Bu proje, multi-modal AI sistemlerinin tek bir platformda nasıl birleşebileceğini göstermek amacıyla geliştirilmiştir.
