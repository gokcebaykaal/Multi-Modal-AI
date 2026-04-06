from pydantic import BaseModel
from typing import Any, Optional


class ApiResponse(BaseModel):
    success: bool = True
    message: str = "İşlem başarılı."
    data: Optional[Any] = None
    error: Optional[Any] = None