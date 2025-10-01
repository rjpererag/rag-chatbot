# PHASE 2: RETRIEVAL AND GENERATION
import os
import logging
from typing import Tuple, Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    GENERATION_MODEL,
    TOP_K_RETRIEVAL
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


SYSTEM_PROMPT = (
    "You are an accurate and helpful corporate assistant. "
    "Your task is to answer the user's question based *ONLY* on the provided context."
    "If the context does not contain the answer, you MUST state: 'I apologize, but I cannot find the answer to this question in the provided documents.'"
    "Do not use any external knowledge. Maintain a professional and polite tone."
)


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(
            "CONTEXT:\n---\n{context}\n---\n\nQUESTION: {question}"
        )
    ]
)


def initialize_rag_components(vector_db_path: str) -> Tuple[Any, ChatOpenAI]:

    logging.info("Initializing RAG components...")

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})

        llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0.1)
        logging.info("RAG components initialized successfully.")
        return retriever, llm

    except Exception as e:
        logging.error(f"Error initializing RAG components: {e}")
        logging.error("Ensure 'faiss_index' exists and OPENAI_API_KEY is set.")
        raise


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(retriever, llm):
    logging.info("Creating the RAG processing chain...")

    rag_chain = (
            {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


def get_answer(rag_chain, user_question: str) -> str:
    logging.info(f"Processing question: '{user_question}'")

    try:
        # Invoke the chain with the user question
        response = rag_chain.invoke(user_question)
        logging.info("Response generated.")
        return response
    except Exception as e:
        logging.error(f"Error during chain execution: {e}")
        return "An internal error occurred during the response generation."


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("CRITICAL: OPENAI_API_KEY not found. Cannot run chatbot.")
    elif not os.path.exists(VECTOR_DB_PATH):
        logging.error(f"CRITICAL: Vector store not found at '{VECTOR_DB_PATH}'. Run 'ingest_data.py' first.")
    else:
        try:
            retriever, llm = initialize_rag_components(VECTOR_DB_PATH)

            rag_chain = create_rag_chain(retriever, llm)

            print("\n--- RAG Chatbot CLI (Type 'exit' to quit) ---")
            while True:
                question = input("You: ")
                if question.lower() == 'exit':
                    break

                answer = get_answer(rag_chain, question)
                print(f"\nAI: {answer}\n")

        except Exception as e:
            # Catch exceptions from initialization to prevent application crash
            logging.error(f"Chatbot failed to start due to an initialization error.")