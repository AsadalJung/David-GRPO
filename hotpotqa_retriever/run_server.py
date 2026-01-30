#!/usr/bin/env python3
"""
Fast Retrieval Server Runner
Simple server execution script
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def find_files():
    """Find required files in the current directory."""
    current_dir = Path(__file__).parent
    
    # Find index files (.faiss extension)
    index_files = list(current_dir.glob("*.faiss"))
    if not index_files:
        print("❌ Cannot find FAISS index file!")
        print("   Please run 'python create_faiss_index.py' first to create the index.")
        return None, None
    
    index_path = index_files[0]
    
    # Find corpus files
    corpus_files = list(current_dir.glob("*corpus*.json"))
    if not corpus_files:
        print("❌ Cannot find corpus file!")
        print("   hotpotqa_corpus.json file is required.")
        return None, None
    
    corpus_path = corpus_files[0]
    
    return str(index_path), str(corpus_path)

def main():
    parser = argparse.ArgumentParser(description="Fast Retrieval Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8001, help="Port number")
    parser.add_argument("--model", default="intfloat/e5-base-v2", help="Model name")
    parser.add_argument("--model-device", default="cuda", help="Model device (cuda/cpu)")
    parser.add_argument("--index-device", default="cpu", help="FAISS index device (cuda/cpu)")
    parser.add_argument("--index-path", help="Index file path (auto-detected if not provided)")
    parser.add_argument("--corpus-path", help="Corpus file path (auto-detected if not provided)")
    
    args = parser.parse_args()
    
    # Auto-detect file paths
    if not args.index_path or not args.corpus_path:
        print("🔍 Auto-detecting files...")
        auto_index, auto_corpus = find_files()
        if not auto_index or not auto_corpus:
            sys.exit(1)
        
        args.index_path = args.index_path or auto_index
        args.corpus_path = args.corpus_path or auto_corpus
    
    # Check file existence
    if not os.path.exists(args.index_path):
        print(f"❌ Cannot find index file: {args.index_path}")
        sys.exit(1)
    
    if not os.path.exists(args.corpus_path):
        print(f"❌ Cannot find corpus file: {args.corpus_path}")
        sys.exit(1)
    
    print("🚀 Starting Fast Retrieval Server...")
    print(f"   📁 Index: {args.index_path}")
    print(f"   📄 Corpus: {args.corpus_path}")
    print(f"   🖥️  Host: {args.host}:{args.port}")
    print(f"   🤖 Model: {args.model}")
    print(f"   ⚡ Model Device: {args.model_device}")
    print(f"   🔍 Index Device: {args.index_device}")
    print()
    
    # Set environment variables
    os.environ["INDEX_PATH"] = args.index_path
    os.environ["CORPUS_PATH"] = args.corpus_path
    os.environ["RETRIEVAL_MODEL"] = args.model
    os.environ["DEVICE"] = args.model_device  # Use model device as default DEVICE
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    
    # Start server
    try:
        import uvicorn
        from fast_retrieval_server import app
        
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except ImportError:
        print("❌ Cannot find uvicorn. Please run 'pip install uvicorn'.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
