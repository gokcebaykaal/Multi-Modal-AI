import json
import hashlib
import logging
from app.core.redis_client import get_redis_client

logger = logging.getLogger(__name__)


def make_cache_key(prefix: str, payload: str) -> str:
    hashed = hashlib.md5(payload.encode("utf-8")).hexdigest()
    return f"{prefix}:{hashed}"


def get_cache(key: str):
    client = get_redis_client()
    if not client:
        return None

    try:
        cached_data = client.get(key)
        if cached_data:
            logger.info(f"Cache HIT: {key}")
            return json.loads(cached_data)

        logger.info(f"Cache MISS: {key}")
        return None

    except Exception as e:
        logger.exception(f"Cache okuma hatası: {str(e)}")
        return None


def set_cache(key: str, data, expire: int = 3600):
    client = get_redis_client()
    if not client:
        return

    try:
        client.setex(key, expire, json.dumps(data, ensure_ascii=False))
        logger.info(f"Cache SET: {key} | expire={expire}")
    except Exception as e:
        logger.exception(f"Cache yazma hatası: {str(e)}")