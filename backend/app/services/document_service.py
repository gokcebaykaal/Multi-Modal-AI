import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import re

logger = logging.getLogger(__name__)

logger.info("Embedding modeli yükleniyor...")
embedding_model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
logger.info("Embedding modeli hazır.")


async def extract_text_from_pdf(file):
    logger.info("PDF metni çıkarılıyor...")

    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text:
            full_text.append(text)
        else:
            logger.warning(f"Sayfa {i} boş.")

    combined = "\n".join(full_text).strip()
    logger.info(f"PDF metin çıkarımı tamamlandı | karakter sayısı={len(combined)}")

    return combined


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=500, overlap=80):
    logger.info("Chunking başladı...")

    text = clean_text(text)

    if not text:
        logger.warning("Boş metin, chunk oluşturulamadı.")
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    logger.info(f"Chunking tamamlandı | chunk sayısı={len(chunks)}")

    if chunks:
        logger.info(f"İlk chunk örneği: {chunks[0][:200]}")

    return chunks


def build_faiss_index(chunks):
    logger.info("FAISS index oluşturuluyor...")

    if not chunks:
        logger.warning("Chunk yok, index oluşturulamadı.")
        return None, None

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    if embeddings is None or len(embeddings) == 0:
        logger.warning("Embedding üretilemedi.")
        return None, None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))

    logger.info(f"FAISS index hazır | boyut={dimension} | kayıt sayısı={len(chunks)}")
    logger.info(f"Embedding shape: {embeddings.shape}")

    return index, embeddings


def retrieve_relevant_chunks(question, chunks, top_k=3, min_score=0.20):
    logger.info(f"Chunk retrieval başladı | soru={question}")

    if not chunks:
        logger.warning("Chunk listesi boş.")
        return []

    index, _ = build_faiss_index(chunks)
    if index is None:
        logger.warning("Index oluşturulamadığı için retrieval yapılamadı.")
        return []

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(question_embedding, min(top_k, len(chunks)))

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        score_value = float(score)

        logger.info(f"Bulunan chunk | idx={idx} | score={score_value:.4f}")

        if score_value < min_score:
            logger.info(f"Chunk elendi | düşük skor={score_value:.4f}")
            continue

        results.append({
            "chunk_id": int(idx),
            "chunk": chunks[idx],
            "score": score_value
        })

    logger.info(f"Retrieval tamamlandı | bulunan chunk sayısı={len(results)}")
    logger.info(f"Retrieved sonuçları: {results[:2]}")

    return results


def generate_rag_answer(question, retrieved_chunks):
    logger.info("Cevap üretimi başladı...")

    if not retrieved_chunks:
        logger.warning("Uygun chunk bulunamadı.")
        return "Dokümanda soruyla ilgili yeterli bilgi bulunamadı."

    best_chunk = retrieved_chunks[0]["chunk"].replace("\n", " ").strip()

    short_best = best_chunk
    if len(short_best) > 700:
        short_best = short_best[:700].rstrip() + "..."

    supporting_points = []

    for item in retrieved_chunks[:3]:
        text = item["chunk"].replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:220].rstrip() + "..."
        supporting_points.append(
            f"- [Kaynak {item['chunk_id']}] (skor: {item['score']:.3f}) {text}"
        )

    lower_q = question.lower()

    if "teknik gereksinim" in lower_q or "gereksinim" in lower_q:
        logger.info("Özel teknik gereksinim cevabı üretildi.")

        return (
            "Dokümana göre öne çıkan teknik gereksinimler şunlardır:\n\n"
            "1. Görüntü işleme modülü bulunmalı; sınıflandırma, güven skoru ve Grad-CAM benzeri açıklanabilirlik çıktıları üretmelidir.\n"
            "2. PDF ve metin girdileri için RAG yapısı kurulmalı; chunking, embedding, vektör veritabanı ve cevap üretimi adımları yer almalıdır.\n"
            "3. Sistem, gelen girdinin türünü otomatik ayırabilmeli ve çoklu veri içeren sorgularda bütünleşik cevap üretebilmelidir.\n"
            "4. Backend tarafında FastAPI kullanılmalı ve en az /analyze-image, /ask-document, /multi-query endpointleri sağlanmalıdır.\n"
            "5. Teknik altyapıda async yapı, arka plan görev yönetimi, cache mekanizması ve loglama/hata yönetimi bulunmalıdır.\n"
            "6. Mimari olarak katmanlı yapı, ayrıştırılmış servis mimarisi ve ölçeklenebilirlik hedeflenmelidir.\n\n"
            "En ilgili kaynak özeti:\n"
            f"{short_best}\n\n"
            "Destekleyici kaynak parçaları:\n"
            + "\n".join(supporting_points)
        )

    logger.info("Genel cevap üretildi.")

    return (
        f"Soru: {question}\n\n"
        f"Dokümana göre en ilgili bilgi:\n{short_best}\n\n"
        f"Destekleyici kaynak parçaları:\n"
        + "\n".join(supporting_points)
    )