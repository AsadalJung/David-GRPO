import os
import logging
import numpy as np
import torch
import faiss
from typing import List, Union, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import time

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Document(BaseModel):
    id: str
    contents: str

class SingleRetrieveResponseWithScore(BaseModel):
    documents: List[Document]
    scores: List[float]

class SingleRetrieveResponseNoScore(BaseModel):
    documents: List[Document]

class RetrieveRequest(BaseModel):
    query: Union[str, List[str]]
    tok_k: int = 3
    return_score: bool = True

class FastRetriever:
    """Fast embedding-based retriever."""
    
    def __init__(self, 
                 model_name: str = "intfloat/e5-base-v2",
                 index_path: str = None,
                 corpus_path: str = None,
                 model_device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 index_device: str = "cpu"):  # Use CPU for FAISS index by default
        
        self.model_device = model_device
        self.index_device = index_device
        self.model_name = model_name
        
        # Memory-efficient model loading
        if model_device == "cuda":
            # Reduce GPU memory usage
            torch.cuda.empty_cache()
            self.model = SentenceTransformer(model_name, device=model_device, model_kwargs={"torch_dtype": torch.bfloat16})
        else:
            self.model = SentenceTransformer(model_name, device=model_device)
        
        # Load index and corpus
        if index_path and os.path.exists(index_path):
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            # GPU index setup
            if index_device == "cuda" and torch.cuda.is_available():
                logger.info("Moving FAISS index to GPU for faster search")
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception as e:
                    logger.warning(f"Failed to move index to GPU, using CPU: {e}")
            else:
                logger.info("Using CPU index")
        else:
            logger.warning("No index path provided or file doesn't exist")
            self.index = None
            
        if corpus_path and os.path.exists(corpus_path):
            logger.info(f"Loading corpus from {corpus_path}")
            with open(corpus_path, 'r', encoding='utf-8') as f:
                self.corpus = json.load(f)
        else:
            logger.warning("No corpus path provided or file doesn't exist")
            self.corpus = []
    
    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries into embeddings."""
        # For E5 models, add the query prefix
        if "e5" in self.model_name.lower():
            queries = [f"query: {q}" for q in queries]
        
        # Generate embeddings
        embeddings = self.model.encode(queries, convert_to_tensor=False, normalize_embeddings=True)
        
        return np.array(embeddings).astype('float32')
    
    def search(self, query: str, num: int = 3, return_score: bool = True):
        """Single-query search."""
        if self.index is None or not self.corpus:
            logger.error("Index or corpus not loaded")
            return None
            
        query_embedding = self._encode_queries([query])
        scores, indices = self.index.search(query_embedding, num)
        
        documents = []
        for idx in indices[0]:
            if idx < len(self.corpus):
                doc = self.corpus[idx]
                documents.append({
                    "id": doc.get("id", str(idx)),
                    "contents": doc.get("contents", doc.get("text", ""))
                })
        
        if return_score:
            return documents, scores[0].tolist()
        else:
            return documents
    
    def batch_search(self, queries: List[str], num: int = 3, return_score: bool = True):
        """Batch query search."""
        if self.index is None or not self.corpus:
            logger.error("Index or corpus not loaded")
            return None
            
        query_embeddings = self._encode_queries(queries)
        scores, indices = self.index.search(query_embeddings, num)
        
        all_documents = []
        all_scores = []
        
        for i, query_indices in enumerate(indices):
            documents = []
            for idx in query_indices:
                if idx < len(self.corpus):
                    doc = self.corpus[idx]
                    documents.append({
                        "id": doc.get("id", str(idx)),
                        "contents": doc.get("contents", doc.get("text", ""))
                    })
            
            all_documents.append(documents)
            all_scores.append(scores[i].tolist())
        
        if return_score:
            return all_documents, all_scores
        else:
            return all_documents

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Fast Retrieval API Service Starting...")
    try:
        # Read paths from config/environment variables
        base_dir = Path(__file__).parent
        default_index = str(base_dir / "hotpotqa_index.faiss")
        default_corpus = str(base_dir / "hotpotqa_corpus.json")

        model_name = os.getenv("RETRIEVAL_MODEL", "intfloat/e5-base-v2")
        index_path = os.getenv("INDEX_PATH", default_index)
        corpus_path = os.getenv("CORPUS_PATH", default_corpus)
        device = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using model: {model_name}")
        logger.info(f"Using device: {device}")
        logger.info(f"Index path: {index_path}")
        logger.info(f"Corpus path: {corpus_path}")
        
        app.state.retriever = FastRetriever(
            model_name=model_name,
            index_path=index_path,
            corpus_path=corpus_path,
            model_device=device,
            index_device="cpu"  # Use CPU for FAISS index
        )
        logger.info("Fast Retrieval API Service Ready!")
    except Exception as e:
        logger.error(f"Failed to initialize fast retriever: {str(e)}")
        raise
    yield
    logger.info("Fast Retrieval API Service Shutting Down...")

app = FastAPI(
    title="Fast Document Retrieval API", 
    description="Fast API for retrieving documents based on queries using GPU acceleration.", 
    lifespan=lifespan
)

@app.post("/retrieve", 
          response_model=Union[SingleRetrieveResponseWithScore, SingleRetrieveResponseNoScore, 
                              List[SingleRetrieveResponseWithScore], List[SingleRetrieveResponseNoScore]])
async def retrieve_docs_endpoint(request: RetrieveRequest):
    try:
        start_time = time.time()
        query = request.query
        tok_k = request.tok_k
        return_score = request.return_score
        retriever = app.state.retriever

        if isinstance(query, str):
            retrieved_result = retriever.search(query=query, num=tok_k, return_score=return_score)
            if retrieved_result is None:
                raise HTTPException(status_code=500, detail="Retrieval failed")
                
            if return_score:
                documents, scores = retrieved_result
                result = SingleRetrieveResponseWithScore(
                    documents=[Document(**doc) for doc in documents],
                    scores=scores
                )
            else:
                documents = retrieved_result
                result = SingleRetrieveResponseNoScore(
                    documents=[Document(**doc) for doc in documents]
                )
                
            elapsed_time = time.time() - start_time
            logger.info(f"Single query processed in {elapsed_time:.3f}s")
            return result
            
        elif isinstance(query, list):
            retrieved_results = retriever.batch_search(queries=query, num=tok_k, return_score=return_score)
            if retrieved_results is None:
                raise HTTPException(status_code=500, detail="Batch retrieval failed")
                
            if return_score:
                docs_list, scores_list = retrieved_results
                result = [
                    SingleRetrieveResponseWithScore(
                        documents=[Document(**doc) for doc in docs],
                        scores=scores
                    )
                    for docs, scores in zip(docs_list, scores_list)
                ]
            else:
                docs_list = retrieved_results
                result = [
                    SingleRetrieveResponseNoScore(
                        documents=[Document(**doc) for doc in docs]
                    )
                    for docs in docs_list
                ]
                
            elapsed_time = time.time() - start_time
            logger.info(f"Batch query ({len(query)} queries) processed in {elapsed_time:.3f}s")
            return result
        else:
            raise ValueError("Query must be a string or a list of strings.")
            
    except ValueError as ve:
        logger.error(f"ValueError occurred: {str(ve)}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "fast-retrieval-api"}

@app.get("/info")
async def service_info():
    """Service info endpoint."""
    retriever = app.state.retriever
    return {
        "model": retriever.model.model_name if retriever.model else "Unknown",
        "device": retriever.model_device,
        "corpus_size": len(retriever.corpus) if retriever.corpus else 0,
        "index_loaded": retriever.index is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    # Read settings from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8001"))  # Default to 8001 to avoid conflicts
    
    logger.info(f"Starting Fast Retrieval API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info") 
