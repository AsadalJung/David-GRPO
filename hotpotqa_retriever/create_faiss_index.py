#!/usr/bin/env python3
"""
FAISS index creation script

Build a FAISS index by embedding the HotpotQA corpus.
"""

import json
import os
import argparse
import numpy as np
import faiss
from typing import List, Dict
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_corpus(corpus_path: str) -> List[Dict]:
    """Load the corpus file."""
    logger.info(f"Loading corpus from {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    logger.info(f"Loaded {len(corpus)} documents")
    return corpus

def create_embeddings(corpus: List[Dict], model_name: str = "intfloat/e5-base-v2", 
                     batch_size: int = 32, device: str = "cuda") -> np.ndarray:
    """Create embeddings from the corpus."""
    logger.info(f"Creating embeddings using model: {model_name}")
    logger.info(f"Device: {device}, Batch size: {batch_size}")
    
    # Load model
    model = SentenceTransformer(model_name, device=device)
    
    # Extract texts
    texts = []
    for doc in tqdm(corpus, desc="Extracting texts"):
        content = doc['contents']
        # For E5 models, add passage prefix
        if "e5" in model_name.lower():
            content = f"passage: {content}"
        texts.append(content)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=False,
        normalize_embeddings=True
    )
    
    # Convert to numpy array
    embeddings = np.array(embeddings).astype('float32')
    logger.info(f"Generated embeddings shape: {embeddings.shape}")
    
    return embeddings

def create_faiss_index(embeddings: np.ndarray, use_gpu: bool = True) -> faiss.Index:
    """Create a FAISS index."""
    logger.info("Creating FAISS index...")
    
    dimension = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dimension}")
    
    # Use CPU index for compatibility when saving
    logger.info("Using CPU for FAISS index (for better compatibility)")
    index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity with normalized vectors)
    
    # Add embeddings
    logger.info("Adding embeddings to index...")
    index.add(embeddings)
    
    logger.info(f"Index created with {index.ntotal} vectors")
    return index

def save_index(index: faiss.Index, output_path: str):
    """Save the FAISS index."""
    logger.info(f"Saving index to {output_path}")
    
    # Directly save CPU index
    faiss.write_index(index, output_path)
    logger.info(f"Index saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create FAISS index from HotpotQA corpus')
    parser.add_argument('--corpus', default='hotpotqa_corpus.json',
                       help='Path to corpus JSON file')
    parser.add_argument('--output', default='hotpotqa_index.faiss',
                       help='Output FAISS index file')
    parser.add_argument('--model', default='intfloat/e5-base-v2',
                       help='Sentence transformer model name')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding generation')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--no-gpu-index', action='store_true',
                       help='Use CPU for FAISS index (default: GPU if available)')
    parser.add_argument('--sample', type=int,
                       help='Use only N documents for testing')
    
    args = parser.parse_args()
    
    try:
        # Load corpus
        corpus = load_corpus(args.corpus)
        
        # Sampling (for testing)
        if args.sample:
            logger.info(f"Using only {args.sample} documents for testing")
            corpus = corpus[:args.sample]
        
        # Generate embeddings
        embeddings = create_embeddings(
            corpus, 
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device
        )
        
        # Create FAISS index
        use_gpu = not args.no_gpu_index
        index = create_faiss_index(embeddings, use_gpu=use_gpu)
        
        # Save index
        save_index(index, args.output)
        
        logger.info("FAISS index creation completed successfully!")
        logger.info(f"Index statistics:")
        logger.info(f"  - Total vectors: {index.ntotal}")
        logger.info(f"  - Dimension: {embeddings.shape[1]}")
        logger.info(f"  - Index file: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during index creation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
