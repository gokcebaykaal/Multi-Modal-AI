from app.services.cv_service import analyze_image
from app.services.rag_service import ask_document

async def multi_query(file, question):
    if file.content_type.startswith("image"):
        return await analyze_image(file)
    else:
        return await ask_document(file, question)