# embedding_server.py
import os
import logging
import shutil # For removing directories
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uvicorn

# --- Dependencies Note ---
# This server uses ONNX and quantization features which require additional dependencies.
# Install them using:
# pip install "sentence-transformers[onnxruntime]>=2.3.0"
# This will install sentence-transformers, optimum, onnx, and onnxruntime.

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# --- Initialize Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s][%(lineno)d] %(message)s')
logging.getLogger('sentence_transformers').setLevel(logging.WARNING) # Reduce sentence_transformers noise
logging.getLogger('onnxruntime').setLevel(logging.WARNING) # Reduce onnxruntime noise unless error
logger = logging.getLogger(__name__)

# --- Environment Variables ---
EMBEDDING_MODEL_NAME_FROM_ENV = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_SERVER_PORT = int(os.environ.get("EMBEDDING_SERVER_PORT", 8003))
EMBEDDING_SERVER_HOST = os.environ.get("EMBEDDING_SERVER_HOST", "0.0.0.0")

# ONNX and Quantization Settings
ENABLE_ONNX_EXPORT = os.environ.get("ENABLE_ONNX_EXPORT", "True").lower() == "true"
ENABLE_QUANTIZATION = os.environ.get("ENABLE_QUANTIZATION", "True").lower() == "true"
ONNX_MODEL_BASE_DIR = os.environ.get("ONNX_MODEL_DIR", "./onnx_models")

# --- Global Model Initialization ---
embedding_model_instance: Optional[SentenceTransformer] = None
onnx_model_used: bool = False
quantized_model_used: bool = False

# Attempt to import quantization utility and set a flag
try:
    from sentence_transformers.quantization import quantize_dynamic
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False
    quantize_dynamic = None # Placeholder
    if ENABLE_QUANTIZATION: # Log warning only if user intends to use it
        logger.warning(
            "OPTIMUM_AVAILABLE=False. `optimum` and `onnxruntime` (or their dependencies) "
            "are likely not installed or `sentence-transformers` version is too old. "
            "ONNX quantization will be unavailable. "
            "Install with: pip install \"sentence-transformers[onnxruntime]>=2.3.0\""
        )


def get_model_specific_onnx_paths(model_name_or_path: str):
    """Generates paths for ONNX and quantized ONNX models for a given base model name."""
    # Create a file-system-safe name from the model name or path
    safe_model_name = model_name_or_path.replace("/", "_").replace("\\", "_").replace(":", "_").replace(".", "_")
    model_onnx_parent_dir = os.path.join(ONNX_MODEL_BASE_DIR, safe_model_name)
    
    # SentenceTransformer saves ONNX models into a directory structure (e.g., model_onnx_parent_dir/onnx/)
    onnx_export_path = os.path.join(model_onnx_parent_dir, "onnx")
    quantized_onnx_export_path = os.path.join(model_onnx_parent_dir, "onnx_quantized")
    
    os.makedirs(model_onnx_parent_dir, exist_ok=True) # Ensure the parent directory for this model's ONNX versions exists
    return onnx_export_path, quantized_onnx_export_path

def initialize_embedding_model_on_startup():
    """
    Loads the SentenceTransformer model.
    Prioritizes quantized ONNX, then standard ONNX, then original Hugging Face model.
    Handles export and quantization if models don't exist.
    """
    global embedding_model_instance, onnx_model_used, quantized_model_used

    onnx_export_path, quantized_onnx_export_path = get_model_specific_onnx_paths(EMBEDDING_MODEL_NAME_FROM_ENV)
    loaded_model_description = ""

    # --- Attempt 1: Load existing quantized ONNX model ---
    if ENABLE_ONNX_EXPORT and ENABLE_QUANTIZATION:
        if not OPTIMUM_AVAILABLE:
            logger.warning("Quantization is enabled, but 'optimum' is not available. Skipping loading/creating quantized model.")
        elif os.path.exists(os.path.join(quantized_onnx_export_path, "model.onnx")):
            logger.info(f"Attempting to load pre-existing quantized ONNX model from: '{quantized_onnx_export_path}'")
            try:
                embedding_model_instance = SentenceTransformer(quantized_onnx_export_path)
                onnx_model_used = True
                quantized_model_used = True
                loaded_model_description = f"pre-existing quantized ONNX model from '{quantized_onnx_export_path}'"
            except Exception as e:
                logger.warning(f"Failed to load pre-existing quantized ONNX model from '{quantized_onnx_export_path}': {e}. "
                               "It might be corrupted. Will attempt to recreate or use other versions.")
                if os.path.exists(quantized_onnx_export_path): # Remove corrupted directory
                    shutil.rmtree(quantized_onnx_export_path)


    # --- Attempt 2: Load existing standard ONNX model (and quantize if enabled and not already loaded) ---
    if not embedding_model_instance and ENABLE_ONNX_EXPORT:
        if os.path.exists(os.path.join(onnx_export_path, "model.onnx")):
            logger.info(f"Attempting to use pre-existing standard ONNX model from: '{onnx_export_path}'")
            try:
                if ENABLE_QUANTIZATION and OPTIMUM_AVAILABLE:
                    logger.info(f"Quantization enabled. Will quantize '{onnx_export_path}' to '{quantized_onnx_export_path}'.")
                    if os.path.exists(quantized_onnx_export_path): # Clean up if exists but failed to load previously
                         shutil.rmtree(quantized_onnx_export_path)
                    os.makedirs(quantized_onnx_export_path, exist_ok=True)
                    
                    quantize_dynamic(onnx_export_path, quantized_onnx_export_path, use_external_data_format=False)
                    logger.info(f"Successfully quantized. Loading quantized model from '{quantized_onnx_export_path}'.")
                    embedding_model_instance = SentenceTransformer(quantized_onnx_export_path)
                    onnx_model_used = True
                    quantized_model_used = True
                    loaded_model_description = f"quantized ONNX model (from existing ONNX) at '{quantized_onnx_export_path}'"
                else: # Use the standard (non-quantized) ONNX model
                    if ENABLE_QUANTIZATION and not OPTIMUM_AVAILABLE:
                        logger.warning("Quantization enabled but optimum not available. Loading standard ONNX.")
                    logger.info(f"Loading standard ONNX model from '{onnx_export_path}'.")
                    embedding_model_instance = SentenceTransformer(onnx_export_path)
                    onnx_model_used = True
                    quantized_model_used = False
                    loaded_model_description = f"pre-existing standard ONNX model from '{onnx_export_path}'"
            except Exception as e:
                logger.warning(f"Failed to load or quantize pre-existing standard ONNX model from '{onnx_export_path}': {e}. "
                               "It might be corrupted. Will attempt to recreate or use other versions.")
                if os.path.exists(onnx_export_path): shutil.rmtree(onnx_export_path) # Remove corrupted
                if os.path.exists(quantized_onnx_export_path): shutil.rmtree(quantized_onnx_export_path) # Clean up attempted quant dir


    # --- Attempt 3: Create ONNX model from Hugging Face (and quantize if enabled and not already loaded) ---
    if not embedding_model_instance and ENABLE_ONNX_EXPORT:
        logger.info(f"No suitable pre-existing ONNX model found/loaded. Attempting to create from '{EMBEDDING_MODEL_NAME_FROM_ENV}'.")
        try:
            logger.info(f"Loading original model '{EMBEDDING_MODEL_NAME_FROM_ENV}' for ONNX export. This may take some time...")
            original_model = SentenceTransformer(EMBEDDING_MODEL_NAME_FROM_ENV)
            
            if os.path.exists(onnx_export_path): shutil.rmtree(onnx_export_path) # Clean up previous attempt
            os.makedirs(onnx_export_path, exist_ok=True)
            logger.info(f"Exporting original model to standard ONNX format at '{onnx_export_path}'. This may take some time...")
            original_model.save_to_onnx(onnx_export_path)
            logger.info(f"Successfully exported to standard ONNX format at '{onnx_export_path}'.")

            if ENABLE_QUANTIZATION and OPTIMUM_AVAILABLE:
                logger.info(f"Quantization enabled. Quantizing '{onnx_export_path}' to '{quantized_onnx_export_path}'. This may take some time...")
                if os.path.exists(quantized_onnx_export_path): shutil.rmtree(quantized_onnx_export_path) # Clean up previous attempt
                os.makedirs(quantized_onnx_export_path, exist_ok=True)
                
                quantize_dynamic(onnx_export_path, quantized_onnx_export_path, use_external_data_format=False)
                logger.info(f"Successfully quantized. Loading quantized model from '{quantized_onnx_export_path}'.")
                embedding_model_instance = SentenceTransformer(quantized_onnx_export_path)
                onnx_model_used = True
                quantized_model_used = True
                loaded_model_description = f"newly created and quantized ONNX model from '{quantized_onnx_export_path}'"
            else: # Use the newly created standard ONNX model
                if ENABLE_QUANTIZATION and not OPTIMUM_AVAILABLE:
                    logger.warning("Quantization enabled but optimum not available. Loading newly created standard ONNX.")
                logger.info(f"Loading newly created standard ONNX model from '{onnx_export_path}'.")
                embedding_model_instance = SentenceTransformer(onnx_export_path)
                onnx_model_used = True
                quantized_model_used = False
                loaded_model_description = f"newly created standard ONNX model from '{onnx_export_path}'"
        except Exception as e:
            logger.error(f"Failed during ONNX export or quantization from '{EMBEDDING_MODEL_NAME_FROM_ENV}': {e}. "
                         "Will fall back to original Hugging Face model.", exc_info=True)
            if os.path.exists(onnx_export_path): shutil.rmtree(onnx_export_path) # Clean up failed export
            if os.path.exists(quantized_onnx_export_path): shutil.rmtree(quantized_onnx_export_path) # Clean up failed quantization

    # --- Fallback: Load original Hugging Face model ---
    if not embedding_model_instance:
        if ENABLE_ONNX_EXPORT: # Log only if ONNX was attempted
            logger.warning("All ONNX/quantization attempts failed or were skipped. Falling back to original Hugging Face model.")
        else:
            logger.info(f"ONNX export is disabled. Loading original Hugging Face model '{EMBEDDING_MODEL_NAME_FROM_ENV}'.")
        
        try:
            embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME_FROM_ENV)
            onnx_model_used = False # Explicitly set flags
            quantized_model_used = False
            loaded_model_description = f"original Hugging Face model '{EMBEDDING_MODEL_NAME_FROM_ENV}' (fallback or ONNX disabled)"
            logger.info(f"Successfully loaded original Sentence Transformer model.")
        except Exception as e:
            logger.critical(f"FATAL ERROR: Could not initialize any embedding model ('{EMBEDDING_MODEL_NAME_FROM_ENV}' or ONNX versions): {e}", exc_info=True)
            raise RuntimeError(f"Failed to load any embedding model version for '{EMBEDDING_MODEL_NAME_FROM_ENV}': {e}") from e

    logger.info(f"--- Model Initialization Complete ---")
    logger.info(f"Model Source: {EMBEDDING_MODEL_NAME_FROM_ENV}")
    logger.info(f"Using: {loaded_model_description}")
    logger.info(f"ONNX Active: {onnx_model_used}, Quantized: {quantized_model_used}")
    logger.info(f"Model Instance: {type(embedding_model_instance)}")


# --- FastAPI App Setup ---
app = FastAPI(
    title="Text Embedding API",
    description="A FastAPI server to generate text embeddings using Sentence Transformers, with ONNX and quantization support.",
    version="1.1.0"
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

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="A list of embeddings, where each embedding is a list of floats.")
    model_name: str = Field(..., description="The name of the base model used.")
    model_type_used: str = Field(..., description="Type of model loaded (e.g., original, ONNX, quantized ONNX).")
    message: Optional[str] = Field(None, description="An optional message, e.g., success or warning.")

# --- API Endpoints ---
@app.post("/embed", response_model=EmbeddingResponse)
async def create_text_embeddings(request: EmbeddingRequest):
    """
    Generates embeddings for a list of input texts.
    """
    global embedding_model_instance, onnx_model_used, quantized_model_used
    if not embedding_model_instance:
        logger.error("Embedding model is not initialized or failed to load.")
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not available. The server might be starting up or encountered an issue."
        )

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided in the request.")

    logger.info(f"Received request to embed {len(request.texts)} text(s). First text (first 50 chars): '{request.texts[0][:50]}...'")

    try:
        generated_embeddings = embedding_model_instance.encode(
            request.texts,
            show_progress_bar=False
        )
        embeddings_list = generated_embeddings.tolist()
        
        model_type_str = "original Hugging Face"
        if onnx_model_used and quantized_model_used:
            model_type_str = "quantized ONNX"
        elif onnx_model_used:
            model_type_str = "standard ONNX"

        logger.info(f"Successfully generated {len(embeddings_list)} embeddings using {model_type_str} model '{EMBEDDING_MODEL_NAME_FROM_ENV}'.")
        return EmbeddingResponse(
            embeddings=embeddings_list,
            model_name=EMBEDDING_MODEL_NAME_FROM_ENV,
            model_type_used=model_type_str,
            message=f"Successfully embedded {len(request.texts)} texts."
        )
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while generating embeddings: {str(e)}"
        )

@app.get("/health")
async def health_check_endpoint():
    """
    Provides a health check for the embedding server.
    """
    global embedding_model_instance, onnx_model_used, quantized_model_used
    
    model_status_message = "Embedding server is running"
    status = "ok"

    if embedding_model_instance:
        model_status_message += " and model is loaded."
    else:
        model_status_message += ", but the embedding model is NOT loaded."
        status = "error"
        
    return {
        "status": status,
        "message": model_status_message,
        "model_name_configured": EMBEDDING_MODEL_NAME_FROM_ENV,
        "onnx_support_enabled_in_config": ENABLE_ONNX_EXPORT,
        "quantization_support_enabled_in_config": ENABLE_QUANTIZATION,
        "optimum_library_available": OPTIMUM_AVAILABLE,
        "currently_using_onnx_model": onnx_model_used,
        "currently_using_quantized_model": quantized_model_used
    }

# --- Main execution for Uvicorn ---
if __name__ == "__main__":
    logger.info(f"Starting Text Embedding API server on {EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}")
    logger.info(f"Using base embedding model: {EMBEDDING_MODEL_NAME_FROM_ENV}")
    logger.info(f"ONNX Export/Usage: {'Enabled' if ENABLE_ONNX_EXPORT else 'Disabled'}")
    if ENABLE_ONNX_EXPORT:
        logger.info(f"Quantization: {'Enabled' if ENABLE_QUANTIZATION and OPTIMUM_AVAILABLE else ('Disabled' if not ENABLE_QUANTIZATION else 'Disabled (Optimum/ONNXRuntime not available)')}")
        logger.info(f"ONNX Model Directory: {ONNX_MODEL_BASE_DIR}")
    
    # FastAPI's @app.on_event("startup") will handle the model initialization.
    uvicorn.run(app, host=EMBEDDING_SERVER_HOST, port=EMBEDDING_SERVER_PORT)