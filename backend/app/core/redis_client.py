import redis
import logging
from app.core.settings import settings

logger = logging.getLogger(__name__)

redis_client = None


def get_redis_client():
    global redis_client

    if redis_client is None:
        try:
            redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True
            )
            redis_client.ping()
            logger.info("Redis bağlantısı başarılı.")
        except Exception as e:
            logger.exception(f"Redis bağlantı hatası: {str(e)}")
            redis_client = None

    return redis_client