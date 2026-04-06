from fastapi import APIRouter, UploadFile, File
from app.services.image_service import analyze_uploaded_image

router = APIRouter()

@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        result = analyze_uploaded_image(file_bytes)
        return result
    except Exception as e:
        return {
            "label": None,
            "confidence": None,
            "explanation": f"Hata oluştu: {str(e)}",
            "gradcam": None
        }