import os
import logging
from typing import List
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document


class FileManager:

    def __init__(self):
        pass

    @staticmethod
    def load_pickle(file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")

        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_documents(data_path: str) -> List[Document]:
        try:
            loader = DirectoryLoader(
                path=data_path,
                glob='**/*.pdf',
                loader_cls=PyPDFLoader,
                silent_errors=True # Good for ignoring files that might fail
            )
            documents = loader.load()
            logging.info(f"Successfully loaded {len(documents)} documents.")
            return documents
        except Exception as e:
            logging.error(f"Error during document loading: {e}")
            return []