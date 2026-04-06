from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from app.services.document_service import (
    extract_text_from_pdf,
    chunk_text,
    retrieve_relevant_chunks,
    generate_rag_answer
)
from app.services.image_service import analyze_uploaded_image

router = APIRouter()


@router.post("/multi-query")
async def multi_query(
    question: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        q = (question or "").strip()

        if not file and not q:
            return {
                "mode": "empty",
                "question": question,
                "answer": "Lütfen bir soru yazın veya bir dosya yükleyin.",
                "sources": []
            }

        # SADECE METİN
        if not file and q:
            lower_q = q.lower()

            if "python" in lower_q:
                answer = "Python, yüksek seviyeli, okunabilirliği kolay bir programlama dilidir."
            elif "yapay zeka" in lower_q:
                answer = "Yapay zeka, makinelerin insan benzeri düşünme ve öğrenme yeteneği kazanmasını sağlayan bir alandır."
            elif "merhaba" in lower_q:
                answer = "Merhaba! Size nasıl yardımcı olabilirim?"
            else:
                answer = f"Sorunuz alındı: '{q}'. Bu demo sürümde sınırlı cevap verilmektedir."

            return {
                "mode": "text",
                "question": q,
                "answer": answer,
                "sources": []
            }

        if file:
            content_type = file.content_type or ""

            # GÖRSEL
            if content_type.startswith("image/"):
                file_bytes = await file.read()
                image_result = analyze_uploaded_image(file_bytes)

                label = image_result.get("label")
                confidence = image_result.get("confidence")
                explanation = image_result.get("explanation")
                gradcam = image_result.get("gradcam")

                if q:
                    answer = (
                        f"Görsel analiz sonucu: {label}. "
                        f"Güven oranı: %{confidence}. "
                        f"Açıklama: {explanation}. "
                        f"Kullanıcı sorusu: {q}"
                    )
                    mode = "image+text"
                else:
                    answer = (
                        f"Görsel analiz sonucu: {label}. "
                        f"Güven oranı: %{confidence}. "
                        f"Açıklama: {explanation}."
                    )
                    mode = "image"

                return {
                    "mode": mode,
                    "question": q,
                    "answer": answer,
                    "sources": [],
                    "file_type": "image",
                    "label": label,
                    "confidence": confidence,
                    "explanation": explanation,
                    "gradcam": gradcam
                }

            # PDF
            if content_type == "application/pdf":
                if not q:
                    return {
                        "mode": "pdf",
                        "question": q,
                        "answer": "PDF yüklendi. Lütfen PDF hakkında bir soru da yazın.",
                        "sources": [],
                        "file_type": "pdf"
                    }

                text = await extract_text_from_pdf(file)
                chunks = chunk_text(text)
                retrieved = retrieve_relevant_chunks(q, chunks, top_k=3)

                if not retrieved:
                    return {
                        "mode": "pdf",
                        "question": q,
                        "answer": "PDF içinde ilgili bilgi bulunamadı.",
                        "sources": [],
                        "file_type": "pdf"
                    }

                answer = generate_rag_answer(q, retrieved)

                return {
                    "mode": "pdf",
                    "question": q,
                    "answer": answer,
                    "sources": retrieved,
                    "file_type": "pdf"
                }

            return {
                "mode": "unsupported",
                "question": q,
                "answer": f"Desteklenmeyen dosya türü: {content_type}",
                "sources": []
            }

    except Exception as e:
        return {
            "mode": "error",
            "question": question,
            "answer": f"Hata oluştu: {str(e)}",
            "sources": []
        }