from langchain.embeddings import HuggingFaceEmbeddings

def get_embedding():
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        # model_kwargs=model_kwargs
        )
    return embeddings