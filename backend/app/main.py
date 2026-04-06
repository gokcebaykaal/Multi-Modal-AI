from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.settings import settings
from app.core.logging_config import setup_logging
from app.core.exceptions import (
    AppException,
    app_exception_handler,
    general_exception_handler,
)
from app.core.middleware import RequestContextMiddleware

from app.routes.image import router as image_router
from app.routes.document import router as document_router
from app.routes.multi import router as multi_router

setup_logging(settings.LOG_LEVEL)

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
)

app.add_middleware(RequestContextMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.include_router(image_router, tags=["Image"])
app.include_router(document_router, tags=["Document"])
app.include_router(multi_router, tags=["Multi"])


@app.get("/")
async def root():
    return {
        "success": True,
        "message": f"{settings.APP_NAME} API çalışıyor."
    }


@app.get("/health")
async def health():
    return {
        "success": True,
        "status": "ok",
        "environment": settings.APP_ENV,
        "app_name": settings.APP_NAME
    }