# tools/extract.py
from bs4 import BeautifulSoup
from dateutil import parser as dparser

def extract_basic(html: str):
    soup = BeautifulSoup(html, "html.parser")

    # title
    title_node = (
        soup.find("meta", property="og:title") or
        soup.find("meta", attrs={"name": "twitter:title"}) or
        soup.find("title")
    )
    if title_node and getattr(title_node, "has_attr", lambda *_: False)("content"):
        title = title_node.get("content") or ""
    else:
        title = title_node.text.strip() if title_node else ""

    # body
    article = soup.find("article")
    if article:
        body = article.get_text("\n").strip()
    else:
        main = soup.find("main") or soup.body
        paragraphs = (main.find_all("p") if main else soup.find_all("p")) or []
        body = "\n\n".join(p.get_text().strip() for p in paragraphs)

    # published_at
    pub = None
    for meta in soup.find_all("meta"):
        key = (meta.get("property") or meta.get("name") or "").lower()
        if "published" in key or key in ("article:published_time", "og:published_time", "pubdate", "date"):
            try:
                pub = dparser.parse(meta.get("content")).isoformat()
                break
            except Exception:
                pass

    return {"title": title, "body": body, "published_at": pub}
