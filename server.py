# embedding_server.py
import os
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uvicorn

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# --- Initialize Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s][%(lineno)d] %(message)s')
logging.getLogger('sentence_transformers').setLevel(logging.WARNING) # Reduce sentence_transformers noise
logger = logging.getLogger(__name__)

# --- Environment Variables ---
EMBEDDING_MODEL_NAME_FROM_ENV = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
# For the embedding server, we'll use a different port, e.g., 8003,
# to avoid conflict if the main QG server runs on 8002.
EMBEDDING_SERVER_PORT = int(os.environ.get("EMBEDDING_SERVER_PORT", 8003))
EMBEDDING_SERVER_HOST = os.environ.get("EMBEDDING_SERVER_HOST", "0.0.0.0")

# --- Global Model Initialization ---
# This will hold the loaded SentenceTransformer model
embedding_model_instance: Optional[SentenceTransformer] = None

def initialize_embedding_model_on_startup():
    """
    Loads the SentenceTransformer model.
    This function is called during FastAPI startup.
    """
    global embedding_model_instance
    try:
        logger.info(f"Initializing Sentence Transformer model: '{EMBEDDING_MODEL_NAME_FROM_ENV}'...")
        embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME_FROM_ENV)
        logger.info(f"Sentence Transformer model '{EMBEDDING_MODEL_NAME_FROM_ENV}' loaded successfully.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not initialize embedding model '{EMBEDDING_MODEL_NAME_FROM_ENV}': {e}", exc_info=True)
        # If the model fails to load, the server cannot function.
        # Raising an error here will prevent FastAPI from starting up successfully.
        raise RuntimeError(f"Failed to load embedding model '{EMBEDDING_MODEL_NAME_FROM_ENV}': {e}") from e

# --- FastAPI App Setup ---
app = FastAPI(
    title="Text Embedding API",
    description="A FastAPI server to generate text embeddings using Sentence Transformers.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event_handler():
    """
    FastAPI startup event handler.
    """
    logger.info("Embedding server starting up...")
    initialize_embedding_model_on_startup()

# --- Pydantic Models for API Request/Response ---
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="A list of texts to be embedded.", min_items=1)
    # Example of how you might add more options in the future:
    # normalize_embeddings: bool = Field(False, description="Whether to normalize the embeddings.")

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="A list of embeddings, where each embedding is a list of floats.")
    model_name: str = Field(..., description="The name of the model used to generate embeddings.")
    message: Optional[str] = Field(None, description="An optional message, e.g., success or warning.")

# --- API Endpoints ---
@app.post("/embed", response_model=EmbeddingResponse)
async def create_text_embeddings(request: EmbeddingRequest):
    """
    Generates embeddings for a list of input texts.
    """
    global embedding_model_instance
    if not embedding_model_instance:
        logger.error("Embedding model is not initialized or failed to load.")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Embedding model is not available. The server might be starting up or encountered an issue."
        )

    if not request.texts:
        # This case should be caught by Pydantic's min_items=1, but as a safeguard:
        raise HTTPException(status_code=400, detail="No texts provided in the request.")

    logger.info(f"Received request to embed {len(request.texts)} text(s). First text (first 50 chars): '{request.texts[0][:50]}...'")

    try:
        # The client (your main QG server) will be responsible for any text cleaning
        # (like the clean_text_for_embedding function) before sending texts here.
        # This server focuses solely on embedding the provided texts.
        generated_embeddings = embedding_model_instance.encode(
            request.texts,
            show_progress_bar=False  # Keep False for server-side logging
        )
        
        # Convert numpy array to list of lists for JSON serialization
        embeddings_list = generated_embeddings.tolist()
        
        logger.info(f"Successfully generated {len(embeddings_list)} embeddings using model '{EMBEDDING_MODEL_NAME_FROM_ENV}'.")
        return EmbeddingResponse(
            embeddings=embeddings_list,
            model_name=EMBEDDING_MODEL_NAME_FROM_ENV,
            message=f"Successfully embedded {len(request.texts)} texts."
        )
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,  # Internal Server Error
            detail=f"An unexpected error occurred while generating embeddings: {str(e)}"
        )

@app.get("/health")
async def health_check_endpoint():
    """
    Provides a health check for the embedding server.
    """
    global embedding_model_instance
    if embedding_model_instance:
        return {
            "status": "ok",
            "message": "Embedding server is running and model is loaded.",
            "model_name": EMBEDDING_MODEL_NAME_FROM_ENV
        }
    else:
        # This state should ideally not be reached if startup event completes.
        # If it is, it means the model isn't loaded.
        return {
            "status": "error",
            "message": "Embedding server is running, but the embedding model is not loaded.",
            "model_name": None
        }

# --- Main execution for Uvicorn ---
if __name__ == "__main__":
    logger.info(f"Starting Text Embedding API server on {EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}")
    logger.info(f"Using embedding model: {EMBEDDING_MODEL_NAME_FROM_ENV}")
    
    # Ensure the model is loaded before uvicorn starts serving,
    # especially if not relying solely on FastAPI's startup event for critical init.
    # However, for FastAPI, using the @app.on_event("startup") is the standard way.
    # If you run this script directly, the startup event will handle initialization.
    
    uvicorn.run(app, host=EMBEDDING_SERVER_HOST, port=EMBEDDING_SERVER_PORT)