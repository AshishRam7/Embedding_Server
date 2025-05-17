# embedding_server.py
import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import uvicorn
import numpy as np

# ONNX and Quantization related imports
import onnxruntime
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import QuantizationConfig
from onnxruntime.quantization import QuantFormat, QuantType

# --- Configuration ---
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s][%(lineno)d] %(message)s')
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('optimum').setLevel(logging.INFO) # Keep INFO for optimum to see optimization steps
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME_FROM_ENV = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_SERVER_PORT = int(os.environ.get("EMBEDDING_SERVER_PORT", 8003))
EMBEDDING_SERVER_HOST = os.environ.get("EMBEDDING_SERVER_HOST", "0.0.0.0")

# --- Model Paths ---
# Store optimized models in a subdirectory
MODEL_CACHE_DIR = Path(__file__).parent / ".model_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)

# Sanitize model name for path
SAFE_MODEL_NAME = "".join(c if c.isalnum() else "_" for c in EMBEDDING_MODEL_NAME_FROM_ENV)

ONNX_EXPORT_DIR = MODEL_CACHE_DIR / f"{SAFE_MODEL_NAME}_onnx"
QUANTIZED_ONNX_MODEL_DIR = MODEL_CACHE_DIR / f"{SAFE_MODEL_NAME}_quantized_onnx"
QUANTIZED_ONNX_MODEL_FILENAME = "model_quantized.onnx" # Standard name after quantization

# --- Global Model and Tokenizer ---
# These will hold the loaded ONNX Runtime session and tokenizer
ort_session: Optional[onnxruntime.InferenceSession] = None
tokenizer: Optional[AutoTokenizer] = None


def mean_pooling(model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Mean Pooling - Take attention mask into account for correct averaging.
    model_output: (batch_size, sequence_length, hidden_size)
    attention_mask: (batch_size, sequence_length)
    """
    token_embeddings = model_output
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    return sum_embeddings / sum_mask

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings to unit length."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

def optimize_and_load_model():
    """
    Handles the logic to download, export to ONNX, quantize, and load the model.
    This function is called during FastAPI startup.
    """
    global ort_session, tokenizer

    quantized_model_path = QUANTIZED_ONNX_MODEL_DIR / QUANTIZED_ONNX_MODEL_FILENAME

    if quantized_model_path.exists() and (QUANTIZED_ONNX_MODEL_DIR / "tokenizer_config.json").exists():
        logger.info(f"Loading pre-quantized ONNX model from: {quantized_model_path}")
        try:
            ort_session = onnxruntime.InferenceSession(str(quantized_model_path), providers=['CPUExecutionProvider'])
            # Load tokenizer from the quantized model directory (saved during quantization)
            tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_ONNX_MODEL_DIR)
            logger.info("Successfully loaded quantized ONNX model and tokenizer.")
            return
        except Exception as e:
            logger.error(f"Failed to load pre-quantized model from {quantized_model_path}. Error: {e}. Will attempt to re-optimize.", exc_info=True)
            # Clean up potentially corrupted files
            if quantized_model_path.exists(): quantized_model_path.unlink()

    logger.info(f"Optimized model not found or loading failed. Starting optimization process for '{EMBEDDING_MODEL_NAME_FROM_ENV}'...")

    try:
        # 1. Export to ONNX using Optimum
        # `ORTModelForFeatureExtraction` can directly load and export SentenceTransformer models
        # if they are based on Hugging Face Transformers.
        logger.info(f"Exporting '{EMBEDDING_MODEL_NAME_FROM_ENV}' to ONNX format...")
        ONNX_EXPORT_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        # Load the original model and tokenizer, then export.
        # Optimum's `from_pretrained` with `export=True` handles this.
        onnx_model_exporter = ORTModelForFeatureExtraction.from_pretrained(
            EMBEDDING_MODEL_NAME_FROM_ENV,
            export=True, # This tells Optimum to convert the PyTorch model to ONNX
            # cache_dir=MODEL_CACHE_DIR / "hf_cache" # Optional: specify HF cache
        )
        onnx_model_exporter.save_pretrained(ONNX_EXPORT_DIR)
        # Tokenizer is also saved by save_pretrained to ONNX_EXPORT_DIR
        logger.info(f"ONNX model and tokenizer saved to: {ONNX_EXPORT_DIR}")


        # 2. Quantize the ONNX model (Dynamic INT8)
        logger.info(f"Applying INT8 dynamic quantization to ONNX model from {ONNX_EXPORT_DIR}...")
        QUANTIZED_ONNX_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Create a quantizer object from the exported ONNX model
        # The default model file name in the directory is usually 'model.onnx'
        quantizer = ORTQuantizer.from_pretrained(ONNX_EXPORT_DIR, file_name="model.onnx")

        # Create a dynamic quantization configuration for INT8
        # We use QDQ format (Quantize-Dequantize) which is common for ONNX Runtime.
        # For dynamic quantization, `is_static` is False.
        # `weights_dtype=QuantType.QInt8` specifies that weights should be quantized to 8-bit integers.
        # Activations in dynamic quantization are typically quantized on-the-fly.
        dqconfig = QuantizationConfig(
            is_static=False,  # Dynamic quantization
            format=QuantFormat.QDQ,
            optimize_model=True, # Ensure the model is optimized by ONNX Runtime after quantization
            per_channel=False, # Per-tensor weight quantization is common for dynamic
            weights_dtype=QuantType.QInt8,
            # For dynamic, activations_dtype is often not specified or is Float32,
            # as they are quantized during runtime.
            # operators_to_quantize=['MatMul', 'Add', 'Gather'] # Example, often not needed for basic dynamic
        )

        # Perform quantization
        quantizer.quantize(
            save_dir=QUANTIZED_ONNX_MODEL_DIR,
            quantization_config=dqconfig,
            file_suffix="_quantized" # This will create model_quantized.onnx
        )
        logger.info(f"Quantized ONNX model saved to: {QUANTIZED_ONNX_MODEL_DIR}")

        # Also save the tokenizer to the quantized model directory for easy loading
        # The tokenizer was already saved in ONNX_EXPORT_DIR by onnx_model_exporter.save_pretrained
        # We can load it from there and save it to the quantized directory.
        temp_tokenizer = AutoTokenizer.from_pretrained(ONNX_EXPORT_DIR)
        temp_tokenizer.save_pretrained(QUANTIZED_ONNX_MODEL_DIR)
        logger.info(f"Tokenizer copied to {QUANTIZED_ONNX_MODEL_DIR}")


        # 3. Load the quantized model for serving
        logger.info(f"Loading the newly quantized ONNX model from {QUANTIZED_ONNX_MODEL_DIR / QUANTIZED_ONNX_MODEL_FILENAME}")
        ort_session = onnxruntime.InferenceSession(
            str(QUANTIZED_ONNX_MODEL_DIR / QUANTIZED_ONNX_MODEL_FILENAME),
            providers=['CPUExecutionProvider']
        )
        tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_ONNX_MODEL_DIR)
        logger.info("Successfully optimized and loaded quantized ONNX model and tokenizer.")

    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not optimize or load embedding model '{EMBEDDING_MODEL_NAME_FROM_ENV}': {e}", exc_info=True)
        raise RuntimeError(f"Failed to optimize/load embedding model: {e}") from e


# --- FastAPI App Setup ---
app = FastAPI(
    title="Optimized Text Embedding API (ONNX + Quantized)",
    description="A FastAPI server to generate text embeddings using a quantized ONNX Sentence Transformer model.",
    version="1.1.0"
)

# Add CORS if needed (e.g., for browser-based testing tools)
origins = ["*"] # Example: Allow all for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event_handler():
    logger.info("Optimized embedding server starting up...")
    optimize_and_load_model() # This will handle the one-time optimization

# --- Pydantic Models for API Request/Response ---
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="A list of texts to be embedded.", min_items=1)

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str = Field(..., description="The name of the model used (original name).")
    model_type: str = Field("quantized_onnx", description="Type of the model being served.")
    message: Optional[str] = None

# --- API Endpoints ---
@app.post("/embed", response_model=EmbeddingResponse)
async def create_text_embeddings(request: EmbeddingRequest):
    global ort_session, tokenizer
    if not ort_session or not tokenizer:
        logger.error("ONNX session or tokenizer is not initialized.")
        raise HTTPException(
            status_code=503,
            detail="Embedding model is not available. The server might be initializing or encountered an error."
        )

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided in the request.")

    logger.info(f"Received request to embed {len(request.texts)} text(s). First text (first 50 chars): '{request.texts[0][:50]}...'")

    try:
        # Tokenize texts
        # The tokenizer was loaded from the directory where the ONNX model was saved.
        inputs = tokenizer(
            request.texts,
            padding=True,      # Pad to max length in batch
            truncation=True,   # Truncate to model's max input length
            return_tensors="np" # Return NumPy arrays
        )

        # Prepare inputs for ONNX Runtime session
        # The input names must match the ONNX model's input names (usually 'input_ids', 'attention_mask', 'token_type_ids')
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        if 'token_type_ids' in inputs: # Some models don't use token_type_ids
             ort_inputs['token_type_ids'] = inputs['token_type_ids'].astype(np.int64)


        # Run inference
        # The output name depends on how the model was exported.
        # For feature extraction models from Optimum, the output is often a list,
        # with the first element being `last_hidden_state` or `token_embeddings`.
        # You might need to inspect your ONNX model (e.g., with Netron) to confirm output names.
        # For many sentence transformer models, the primary output will be token embeddings.
        model_output_tuple = ort_session.run(None, ort_inputs)
        
        # Assuming the first output is the token embeddings / last_hidden_state
        token_embeddings_np = model_output_tuple[0] # Shape: (batch_size, sequence_length, hidden_size)

        # Perform pooling (e.g., mean pooling)
        sentence_embeddings_np = mean_pooling(token_embeddings_np, inputs['attention_mask'])

        # Normalize embeddings (common for sentence transformers)
        normalized_sentence_embeddings = normalize_embeddings(sentence_embeddings_np)

        embeddings_list = normalized_sentence_embeddings.tolist()

        logger.info(f"Successfully generated {len(embeddings_list)} embeddings using quantized ONNX model '{EMBEDDING_MODEL_NAME_FROM_ENV}'.")
        return EmbeddingResponse(
            embeddings=embeddings_list,
            model_name=EMBEDDING_MODEL_NAME_FROM_ENV, # Report original model name
            model_type="quantized_onnx_int8_dynamic",
            message=f"Successfully embedded {len(request.texts)} texts using optimized model."
        )
    except Exception as e:
        logger.error(f"Error during ONNX embedding generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while generating embeddings with ONNX model: {str(e)}"
        )

@app.get("/health")
async def health_check_endpoint():
    global ort_session, tokenizer
    if ort_session and tokenizer:
        return {
            "status": "ok",
            "message": "Optimized embedding server is running and quantized ONNX model is loaded.",
            "model_name": EMBEDDING_MODEL_NAME_FROM_ENV,
            "model_type": "quantized_onnx_int8_dynamic",
            "quantized_model_path": str(QUANTIZED_ONNX_MODEL_DIR / QUANTIZED_ONNX_MODEL_FILENAME)
        }
    else:
        return {
            "status": "error",
            "message": "Optimized embedding server is running, but the ONNX model/tokenizer is not loaded or initialization failed.",
            "model_name": EMBEDDING_MODEL_NAME_FROM_ENV,
            "model_type": None
        }

# --- Main execution for Uvicorn ---
if __name__ == "__main__":
    logger.info(f"Starting Optimized Text Embedding API server on {EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}")
    logger.info(f"Using embedding model (original): {EMBEDDING_MODEL_NAME_FROM_ENV}")
    logger.info(f"Optimized models will be stored in: {MODEL_CACHE_DIR}")
    
    # The `optimize_and_load_model()` is called by FastAPI's startup event.
    uvicorn.run(app, host=EMBEDDING_SERVER_HOST, port=EMBEDDING_SERVER_PORT)