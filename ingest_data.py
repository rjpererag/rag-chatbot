# PHASE 1: Indexing
import os
import logging
from typing import List, Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


from utils.file_manager import FileManager

from config import (
    DATA_PATH, VECTOR_DB_PATH, CHUNK_SIZE,
    CHUNK_OVERLAP, EMBEDDING_MODEL
)


def split_documents(documents: List[Document]) -> List[Document]:
    logging.info(f"Splitting documents into chunks (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    logging.info(f"Created {len(chunks)} total text chunks.")
    return chunks


def create_vector_store(chunks: List[Document], vector_db_path: str):
    logging.info(f"Initializing OpenAI Embeddings model: {EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    logging.info("Creating FAISS vector store and generating embeddings...")
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(vector_db_path)
        logging.info(f"Vector store successfully created and saved to: {vector_db_path}")

    except Exception as e:
        logging.error(f"Error during vector store creation: {e}")



def main():
    load_dotenv()
    file_manager = FileManager()
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("CRITICAL: OPENAI_API_KEY not found. Please check your .env file.")
        return

    if not os.path.exists(DATA_PATH) or not any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
        logging.warning(f"The '{DATA_PATH}' directory is missing or contains no PDF files. Please add documents.")
        return

    documents = file_manager.load_documents(DATA_PATH)
    documents = documents[3:4]  # USING ONLY FREQUENTLY ASKED QUESTIONS
    if documents:
        chunks = split_documents(documents)
        if chunks:
            create_vector_store(chunks, VECTOR_DB_PATH)


if __name__ == "__main__":
    main()