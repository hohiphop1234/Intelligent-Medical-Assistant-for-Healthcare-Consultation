import sys
import time

def log(msg):
    print(msg, flush=True)

try:
    log("1. Importing fastapi...")
    import fastapi
    log("2. Importing uvicorn...")
    import uvicorn
    log("3. Importing torch...")
    import torch
    log("4. Importing sentence_transformers...")
    import sentence_transformers
    log("5. Importing chromadb...")
    import chromadb
    log("6. Importing LangGraphPipeline...")
    from src.langgraph_pipeline import LangGraphPipeline
    log("7. All imports successful!")
except Exception as e:
    log(f"Exception: {e}")
