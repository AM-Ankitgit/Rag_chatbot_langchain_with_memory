from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from glob import glob
import os
from pathlib import Path
import argparse
from src.rag_components.get_embeddingfile import get_embedding
from src.constant import data_path
from src.rag_components.web_loader import get_webdata



CHROMA_PATH = "chroma"

def load_data(folder_name):
    loaders = {
        '.pdf': PyMuPDFLoader,
        '.xml': UnstructuredXMLLoader,
        '.csv': CSVLoader,
        '.txt': TextLoader,
        }
    
    csv_txt_xml_file = glob(os.path.join(folder_name,"*.*"))
    all_doc = []

    for i in csv_txt_xml_file:
        ext  = Path(i).suffix
        # print(ext)
        doc = DirectoryLoader(
            path=folder_name,
            glob=f"**/*{ext}",
            loader_cls=loaders[ext]).load()
        # print(doc)
        all_doc.extend(doc)

    # print(all_doc)
    # print(len(all_doc))
    return all_doc







def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)



def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma.from_documents(
        chunks,
        embedding=get_embedding(),
        persist_directory=CHROMA_PATH  # if want to save data locally
    )

    # Calculate Page IDs.
    # chunks_with_ids = calculate_chunk_ids(chunks)

    # # Add or Update the documents.
    # existing_items = db.get(include=[])  # IDs are always included by default
    # existing_ids = set(existing_items["ids"])
    # print(f"Number of existing documents in DB: {len(existing_ids)}")

    # # Only add documents that don't exist in the DB.
    # new_chunks = []
    # for chunk in chunks_with_ids:
    #     if chunk.metadata["id"] not in existing_ids:
    #         new_chunks.append(chunk)

    # if len(new_chunks):
    #     print(f"Adding new documents: {len(new_chunks)}")
    #     new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
    #     db.add_documents(new_chunks, ids=new_chunk_ids)
    #     db.persist()
    # else:
    #     print("No new documents to add")
    


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/pdf_name.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks





import shutil

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def main(folder_name):
    # from cli we can clear the database by passing flag as the arg --reset
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset",action="store_true",help="Reset the database")
    args   = parser.parse_args()

    if args.reset:
        print("Database has been cleared")
        clear_database()

    document = load_data(folder_name) # it return list of data
    web_doc = get_webdata()
    all_doc = document+web_doc
    # print(all_doc)
    chunks = split_documents(all_doc)
    add_to_chroma(chunks) # if file chunk vector is already is added to database  then it will not added only newly updated file will updated



if __name__=="__main__":
    folder_name = data_path
    main(folder_name)
