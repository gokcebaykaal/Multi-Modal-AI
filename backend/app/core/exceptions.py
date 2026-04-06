from fastapi import Request
from fastapi.responses import JSONResponse
from starlette import status
import logging

logger = logging.getLogger(__name__)


class AppException(Exception):
    def __init__(self, message: str, status_code: int = 400, error_code: str = "APP_ERROR"):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(message)


async def app_exception_handler(request: Request, exc: AppException):
    logger.warning(f"AppException: {exc.error_code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
            },
        },
    )


async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Beklenmeyen bir hata oluştu.",
            },
        },
    )