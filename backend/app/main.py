from fastapi import FastAPI
from app.routes import image, document, multi
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Multi Modal AI System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image.router)
app.include_router(document.router)
app.include_router(multi.router)

@app.get("/")
def root():
    return {"message": "API çalışıyor 🚀"}