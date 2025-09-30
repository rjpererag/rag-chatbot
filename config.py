# config.py

# --- Directory Settings ---
DATA_PATH = "documents"
VECTOR_DB_PATH = "faiss_index"

# --- Chunking Strategy (Crucial Hyperparameters) ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- Model Settings ---
# The Embedding Model to turn text into vectors
EMBEDDING_MODEL = "text-embedding-3-small"
# The Generative Model (LLM) for answering
GENERATION_MODEL = "gpt-4o"
# K: The number of top-relevant documents to retrieve
TOP_K_RETRIEVAL = 5