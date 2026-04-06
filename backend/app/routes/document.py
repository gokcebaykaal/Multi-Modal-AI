from fastapi import APIRouter, UploadFile, File, Form
from app.services.document_service import (
    extract_text_from_pdf,
    chunk_text,
    retrieve_relevant_chunks,
    generate_rag_answer
)

router = APIRouter()

@router.post("/ask-document")
async def ask_document(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        text = await extract_text_from_pdf(file)
        chunks = chunk_text(text)
        retrieved = retrieve_relevant_chunks(question, chunks, top_k=3)

        if not retrieved:
            return {
                "question": question,
                "answer": "İlgili bilgi bulunamadı.",
                "sources": []
            }

        answer = generate_rag_answer(question, retrieved)

        return {
            "question": question,
            "answer": answer,
            "sources": retrieved
        }

    except Exception as e:
        return {
            "question": question,
            "answer": f"Doküman işlenirken hata oluştu: {str(e)}",
            "sources": []
        }


@router.post("/ask-text")
async def ask_text(question: str = Form(...)):
    try:
        q = question.lower()

        if "python" in q:
            answer = "Python, yüksek seviyeli, okunabilirliği kolay bir programlama dilidir."
        elif "yapay zeka" in q:
            answer = "Yapay zeka, makinelerin insan benzeri düşünme ve öğrenme yeteneği kazanmasını sağlayan bir alandır."
        elif "merhaba" in q:
            answer = "Merhaba! Size nasıl yardımcı olabilirim?"
        else:
            answer = f"Sorunuz alındı: '{question}'. Bu demo sürümde sınırlı cevap verilmektedir."

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:
        return {
            "question": question,
            "answer": f"Hata oluştu: {str(e)}"
        }