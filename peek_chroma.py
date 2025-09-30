# peek_chroma.py (verbose debug, print FULL documents)
import os
import sys
import traceback
import chromadb
from dotenv import load_dotenv


def log(section, msg):
    print(f"[{section}] {msg}")


def dump_dir(path, depth=1, prefix=""):
    try:
        if not os.path.exists(path):
            log("FS", f"경로 없음: {path}")
            return
        for name in os.listdir(path):
            p = os.path.join(path, name)
            if os.path.isdir(p):
                print(f"{prefix}[D] {name}/")
                if depth > 0:
                    dump_dir(p, depth=depth - 1, prefix=prefix + "    ")
            else:
                size = os.path.getsize(p)
                print(f"{prefix}[F] {name} ({size} bytes)")
    except Exception as e:
        log("FS", f"목록 실패: {e}")
        traceback.print_exc()


def dump_collection(client, name, limit=5):
    print("\n=== {} ===".format(name))
    try:
        log("COL", f"get_collection({name}) 시도")
        col = client.get_collection(name=name)
        log("COL", f"get_collection({name}) 성공")
    except Exception as e:
        log("COL", f"{name} 가져오기 실패: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        print("[없음]")
        return

    try:
        c = col.count()
        log("COL", f"{name}.count() = {c}")
        print("count:", c)
    except Exception as e:
        log("COL", f"{name}.count() 실패: {e.__class__.__name__}: {e}")
        traceback.print_exc()

    try:
        log("COL", f"{name}.peek(limit={limit}) 시도")
        peek = col.peek(limit=limit)
        log("COL", f"{name}.peek() keys: {list(peek.keys())}")

        ids = peek.get("ids", []) or []
        metadatas = peek.get("metadatas", []) or []
        documents = peek.get("documents", []) or []

        for i, doc_id in enumerate(ids):
            md = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            print("-" * 80)
            print(f"id: {doc_id}")
            print(f"name: {md.get('name')}")
            print(f"kind: {md.get('kind')}")
            print(f"url: {md.get('url')}")
            print(f"tags: {md.get('tags')}")
            print(f"metadata(all): {md}")  # 원하면 이 줄을 주석 처리
            print("document (FULL):")
            # ↓↓↓ 요약 없이 전체 출력 ↓↓↓
            if isinstance(doc, str):
                print(doc)
            else:
                print(str(doc))
            print("-" * 80)

    except Exception as e:
        log("COL", f"{name}.peek() 실패: {e.__class__.__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # 콘솔 출력 인코딩(윈도우 대응)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 0) 환경/경로 정보
    log("BOOT", "dotenv 로드 시도")
    load_dotenv()
    log("BOOT", "dotenv 로드 완료")

    cwd = os.getcwd()
    abspath = os.path.abspath(".")
    log("ENV", f"python: {sys.version.split()[0]}")
    log("ENV", f"CWD: {cwd}")
    log("ENV", f"ABS . : {abspath}")

    VDB_PATH = os.environ.get("VDB_PATH", "./data/vector_store")
    VDB_ABS = os.path.abspath(VDB_PATH)
    log("ENV", f"VDB_PATH (env): {VDB_PATH}")
    log("ENV", f"VDB_PATH (abs): {VDB_ABS}")

    # 1) 경로 존재/권한 체크 + 파일 트리 스냅샷
    exists = os.path.exists(VDB_PATH)
    is_dir = os.path.isdir(VDB_PATH)
    log("FS", f"exists={exists}, is_dir={is_dir}")
    if exists and is_dir:
        print("\n[FS] VDB_PATH 1-depth 파일/디렉토리 목록:")
        dump_dir(VDB_PATH, depth=1)
    else:
        log("FS", "경로가 없거나 디렉토리가 아님 (처음 실행이라면 정상일 수 있음)")

    # 2) Chroma 클라이언트 생성
    try:
        log("CHROMA", "PersistentClient 생성 시도")
        client = chromadb.PersistentClient(path=VDB_PATH)
        log("CHROMA", "PersistentClient 생성 성공")
    except Exception as e:
        log("CHROMA", f"PersistentClient 생성 실패: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # 3) 전체 컬렉션 목록
    try:
        log("CHROMA", "list_collections() 시도")
        cols = client.list_collections()
        print("=== 모든 컬렉션 ===")
        for c in cols:
            try:
                cnt = c.count()
            except Exception:
                cnt = "?"
            print(f"- {c.name} (count: {cnt})")
        log("CHROMA", f"list_collections() 완료: {len(cols)}개")
    except Exception as e:
        log("CHROMA", f"list_collections 실패: {e.__class__.__name__}: {e}")
        traceback.print_exc()

    # 4) 우리가 주로 쓰는 컬렉션 상세
    cols = {c.name for c in client.list_collections()}
    for name in ["companies", "industries", "investment_ai"]:
        if name in cols:
            dump_collection(client, name)
        else:
            print(f"\n=== {name} ===\n[없음]")
