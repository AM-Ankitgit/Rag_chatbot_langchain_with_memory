from langchain_chroma import Chroma
from src.rag_components.get_embeddingfile import get_embedding
import os

CHROMA_DB_PATH  = 'chroma'
CHROMA_DB_INSTANCE   = None
# IS_USING_IMAGE_RUNTIME = bool(os.environ.get("IS_USING_IMAGE_RUNTIME", False))

def get_chroma_db():
    global CHROMA_DB_INSTANCE
    global CHROMA_DB_PATH

    if not CHROMA_DB_INSTANCE:
        # config for aws

        CHROMA_DB_INSTANCE  = Chroma(persist_directory=CHROMA_DB_PATH,
                                     embedding_function=get_embedding()
                                     )
        print("Chroma db is exited")
    return CHROMA_DB_INSTANCE