from fastapi import APIRouter, UploadFile, File, Form
from app.services.document_service import extract_text_from_pdf, chunk_text, retrieve_relevant_chunks

router = APIRouter()

@router.post("/ask-document")
async def ask_document(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    text = await extract_text_from_pdf(file)
    chunks = chunk_text(text)
    retrieved = retrieve_relevant_chunks(question, chunks, top_k=3)

    if not retrieved:
        return {
            "question": question,
            "answer": "İlgili bilgi bulunamadı.",
            "sources": []
        }

    best_answer = retrieved[0]["chunk"]

    return {
        "question": question,
        "answer": best_answer,
        "sources": retrieved
    }