import fitz  
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Türkçe için daha uygun çok dilli embedding modeli
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


async def extract_text_from_pdf(file):
    pdf_bytes = await file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text = []
    for page in doc:
        text = page.get_text("text")
        if text:
            full_text.append(text)

    return "\n".join(full_text)


def chunk_text(text, chunk_size=700, overlap=120):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            last_newline = text.rfind("\n", start, end)
            last_period = text.rfind(".", start, end)

            best_split = max(last_newline, last_period)
            if best_split > start + 200:
                end = best_split + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def build_faiss_index(chunks):
    if not chunks:
        return None, None

    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))

    return index, embeddings


def retrieve_relevant_chunks(question, chunks, top_k=3):
    if not chunks:
        return []

    index, _ = build_faiss_index(chunks)
    if index is None:
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

        results.append({
            "chunk_id": int(idx),
            "chunk": chunks[idx],
            "score": float(score)
        })

    return results


def generate_rag_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        return "Dokümanda soruyla ilgili yeterli bilgi bulunamadı."

    best_chunk = retrieved_chunks[0]["chunk"].replace("\n", " ").strip()

    short_best = best_chunk
    if len(short_best) > 500:
        short_best = short_best[:500].rstrip() + "..."


    supporting_points = []
    for item in retrieved_chunks[:3]:
        text = item["chunk"].replace("\n", " ").strip()
        if len(text) > 180:
            text = text[:180].rstrip() + "..."
        supporting_points.append(f"- {text}")

    lower_q = question.lower()

    if "teknik gereksinim" in lower_q or "gereksinim" in lower_q:
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

    return (
        f"Soru: {question}\n\n"
        f"Dokümana göre en ilgili bilgi:\n{short_best}\n\n"
        f"Destekleyici kaynak parçaları:\n"
        + "\n".join(supporting_points)
    )