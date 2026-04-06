from sentence_transformers import SentenceTransformer
import faiss
import fitz 

model = SentenceTransformer("all-MiniLM-L6-v2")

async def ask_document(file, question):
    doc = fitz.open(stream=file.file.read(), filetype="pdf")

    text = ""
    for page in doc:
        text += page.get_text()

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    q_embedding = model.encode([question])
    D, I = index.search(q_embedding, k=3)

    context = [chunks[i] for i in I[0]]

    return {
        "answer": "Bu bir örnek cevaptır",
        "sources": context
    }