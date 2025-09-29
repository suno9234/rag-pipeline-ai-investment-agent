# tools/robots.py
import urllib.parse as urlparse
from urllib.robotparser import RobotFileParser
from functools import lru_cache
from tools.utils import user_agent

@lru_cache(maxsize=512)
def _rp_for_root(root: str) -> RobotFileParser:
    robots_url = urlparse.urljoin(root, "/robots.txt")
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        pass
    return rp

def is_allowed(url: str, ua: str | None = None) -> bool:
    ua = ua or user_agent()
    parsed = urlparse.urlparse(url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    rp = _rp_for_root(root)
    try:
        return rp.can_fetch(ua, url)
    except Exception:
        return True
