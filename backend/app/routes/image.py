from fastapi import APIRouter, UploadFile, File
from app.core.exceptions import AppException
from app.services.image_service import analyze_uploaded_image
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze-image")
async def analyze_image_route(file: UploadFile = File(...)):
    if not file.filename:
        raise AppException("Dosya adı bulunamadı.", 400, "FILE_NAME_MISSING")

    allowed_extensions = (".png", ".jpg", ".jpeg", ".webp")
    if not file.filename.lower().endswith(allowed_extensions):
        raise AppException(
            "Desteklenmeyen görsel formatı. PNG, JPG, JPEG veya WEBP yükleyin.",
            400,
            "INVALID_IMAGE_TYPE"
        )

    logger.info(f"Görsel analiz isteği alındı: {file.filename}")

    file_bytes = await file.read()
    result = analyze_uploaded_image(file_bytes)

    if not result:
        raise AppException("Görsel analiz edilemedi.", 500, "IMAGE_ANALYSIS_FAILED")

    logger.info(f"Görsel analiz tamamlandı: {file.filename}")

    return {
        "success": True,
        "message": "Görsel başarıyla analiz edildi.",
        "data": result
    }