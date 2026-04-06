from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
from app.core.exceptions import AppException
from app.core.cache import make_cache_key, get_cache, set_cache
from app.core.settings import settings
from app.services.document_service import (
    extract_text_from_pdf,
    chunk_text,
    retrieve_relevant_chunks,
    generate_rag_answer
)
import hashlib
import logging
import io

router = APIRouter()
logger = logging.getLogger(__name__)


async def background_process_pdf(pdf_bytes: bytes, question: str):
    from fastapi import UploadFile

    logger.info("BACKGROUND: PDF işleme başladı")

    temp_file = UploadFile(filename="temp.pdf", file=io.BytesIO(pdf_bytes))

    text = await extract_text_from_pdf(temp_file)
    chunks = chunk_text(text)
    retrieve_relevant_chunks(question, chunks, top_k=3)

    logger.info("BACKGROUND: PDF işleme tamamlandı")


@router.post("/ask-document")
async def ask_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    question: str = Form(...)
):
    if not file.filename:
        raise AppException("Dosya adı bulunamadı.", 400, "FILE_NAME_MISSING")

    if not file.filename.lower().endswith(".pdf"):
        raise AppException("Lütfen yalnızca PDF dosyası yükleyin.", 400, "INVALID_FILE_TYPE")

    if not question or not question.strip():
        raise AppException("Soru alanı boş bırakılamaz.", 400, "QUESTION_REQUIRED")

    normalized_question = question.strip()
    pdf_bytes = await file.read()
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

    cache_payload = f"{pdf_hash}:{normalized_question}"
    cache_key = make_cache_key("ask_document", cache_payload)

    cached_response = get_cache(cache_key)
    if cached_response:
        return {
            "success": True,
            "message": "Cevap cache üzerinden getirildi.",
            "data": cached_response
        }

    logger.info(f"Doküman soru isteği alındı: file={file.filename}, question={normalized_question}")

    temp_file = UploadFile(filename=file.filename, file=io.BytesIO(pdf_bytes))
    text = await extract_text_from_pdf(temp_file)

    if not text or not text.strip():
        raise AppException("PDF içeriği okunamadı veya boş.", 400, "EMPTY_DOCUMENT")

    chunks = chunk_text(text)

    if not chunks:
        raise AppException("Doküman parçalanamadı.", 500, "CHUNKING_FAILED")

    retrieved = retrieve_relevant_chunks(normalized_question, chunks, top_k=3)

    if not retrieved:
        response_data = {
            "question": normalized_question,
            "answer": "İlgili bilgi bulunamadı.",
            "sources": []
        }

        set_cache(cache_key, response_data, settings.CACHE_EXPIRE_SECONDS)

        return {
            "success": True,
            "message": "İlgili bilgi bulunamadı.",
            "data": response_data
        }

    best_answer = generate_rag_answer(normalized_question, retrieved)

    response_data = {
        "question": normalized_question,
        "answer": best_answer,
        "sources": retrieved
    }

    set_cache(cache_key, response_data, settings.CACHE_EXPIRE_SECONDS)

    background_tasks.add_task(background_process_pdf, pdf_bytes, normalized_question)

    logger.info(f"Doküman cevabı üretildi ve cache'e yazıldı: file={file.filename}")

    return {
        "success": True,
        "message": "Doküman başarıyla analiz edildi.",
        "data": response_data
    }