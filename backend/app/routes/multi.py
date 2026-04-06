from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
import logging
import hashlib
import io

from app.core.exceptions import AppException
from app.core.cache import make_cache_key, get_cache, set_cache
from app.core.settings import settings
from app.services.document_service import (
    extract_text_from_pdf,
    chunk_text,
    retrieve_relevant_chunks,
    generate_rag_answer
)
from app.services.image_service import analyze_uploaded_image

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/multi-query")
async def multi_query(
    question: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    q = (question or "").strip()

    if not file and not q:
        raise AppException(
            "Lütfen bir soru yazın veya bir dosya yükleyin.",
            400,
            "EMPTY_INPUT"
        )

    logger.info(f"Multi-query başladı | question={q} file={file.filename if file else None}")

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

        logger.info("Text mode tamamlandı")

        return {
            "success": True,
            "mode": "text",
            "data": {
                "question": q,
                "answer": answer,
                "sources": []
            }
        }

    if file:
        content_type = file.content_type or ""
        logger.info(f"Dosya tipi: {content_type}")

        if content_type.startswith("image/"):
            file_bytes = await file.read()
            image_result = analyze_uploaded_image(file_bytes)

            if not image_result:
                raise AppException("Görsel analiz edilemedi.", 500, "IMAGE_FAILED")

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

            logger.info("Image mode tamamlandı")

            return {
                "success": True,
                "mode": mode,
                "data": {
                    "question": q,
                    "answer": answer,
                    "sources": [],
                    "file_type": "image",
                    "label": label,
                    "confidence": confidence,
                    "explanation": explanation,
                    "gradcam": gradcam
                }
            }

        if content_type == "application/pdf":
            if not q:
                raise AppException(
                    "PDF ile birlikte bir soru da gönderilmelidir.",
                    400,
                    "QUESTION_REQUIRED"
                )

            pdf_bytes = await file.read()
            pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

            cache_payload = f"{pdf_hash}:{q}"
            cache_key = make_cache_key("multi_pdf", cache_payload)

            cached_response = get_cache(cache_key)
            if cached_response:
                return {
                    "success": True,
                    "mode": "pdf",
                    "message": "Cevap cache üzerinden getirildi.",
                    "data": cached_response
                }

            temp_file = UploadFile(filename=file.filename, file=io.BytesIO(pdf_bytes))

            text = await extract_text_from_pdf(temp_file)

            if not text or not text.strip():
                raise AppException("PDF içeriği okunamadı.", 400, "EMPTY_DOCUMENT")

            chunks = chunk_text(text)

            if not chunks:
                raise AppException("Chunk oluşturulamadı.", 500, "CHUNK_FAILED")

            retrieved = retrieve_relevant_chunks(q, chunks, top_k=3)

            if not retrieved:
                response_data = {
                    "question": q,
                    "answer": "PDF içinde ilgili bilgi bulunamadı.",
                    "sources": [],
                    "file_type": "pdf"
                }
                set_cache(cache_key, response_data, settings.CACHE_EXPIRE_SECONDS)

                return {
                    "success": True,
                    "mode": "pdf",
                    "data": response_data
                }

            answer = generate_rag_answer(q, retrieved)

            response_data = {
                "question": q,
                "answer": answer,
                "sources": retrieved,
                "file_type": "pdf"
            }

            set_cache(cache_key, response_data, settings.CACHE_EXPIRE_SECONDS)

            logger.info("PDF mode tamamlandı ve cache'e yazıldı")

            return {
                "success": True,
                "mode": "pdf",
                "data": response_data
            }

        raise AppException(
            f"Desteklenmeyen dosya türü: {content_type}",
            400,
            "UNSUPPORTED_FILE"
        )