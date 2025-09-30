import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VDB_PATH = os.getenv("VDB_PATH", "./vector_store")

def get_embeddings():
    """
    HuggingFace 로컬 임베딩 모델 사용.
    최초 실행 시 HuggingFace Hub에서 모델 다운로드 후 캐시에 저장,
    이후 실행은 로컬 캐시에서 불러옵니다.
    """
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"}
    )

def get_vector_store():
    """
    Chroma VectorStore 생성/로드.
    """
    embeddings = get_embeddings()
    vectordb = Chroma(
        collection_name="investment_ai",
        embedding_function=embeddings,
        persist_directory=VDB_PATH
    )
    return vectordb