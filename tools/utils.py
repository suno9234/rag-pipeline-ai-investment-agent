# tools/utils.py
from tenacity import retry, wait_exponential, stop_after_attempt
import random
import time

DEFAULT_UA = "Mozilla/5.0 (compatible; SkalaBot/1.0; +https://example.com/bot)"


def user_agent() -> str:
    return DEFAULT_UA


def polite_sleep(min_ms=200, max_ms=700):
    time.sleep(random.uniform(min_ms / 1000, max_ms / 1000))


retry_policy = dict(
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(3),
    reraise=True,
)
