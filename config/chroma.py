import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VDB_PATH = os.getenv("VDB_PATH", "./data/vector_store")
MODEL_NAME = os.getenv("EMBED_MODEL", "jhgan/ko-sroberta-multitask")  # 한글 임베딩

def get_embeddings():
    """
    HuggingFace 로컬 임베딩 모델 사용.
    최초 1회 다운로드 후 캐시 재사용.
    """
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )

def _get_store(collection_name: str) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=VDB_PATH
    )

def get_company_store() -> Chroma:
    """회사(기업) 정보 전용 컬렉션"""
    return _get_store("companies")

def get_industry_store() -> Chroma:
    """산업 보고서 전용 컬렉션"""
    return _get_store("industries")
