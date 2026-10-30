#!/usr/bin/env python3
"""
Retrieval Pipeline for SIGGRAPH 2025 Papers.

Implements hybrid search:
1. Semantic search (embeddings via OpenRouter + Qdrant Cloud)
2. Keyword search (BM25 - runs locally)
3. Reranking (Cohere API - optional)

Usage:
    from retrieval_pipeline import RetrievalPipeline
    
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve("3D Gaussian Splatting", top_k=5)
"""

import json
import logging
import os
import re
import requests
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
load_dotenv()

# Must match the collection name used in upload_to_qdrant.py
COLLECTION_NAME = "siggraph2025_papers"
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Represents a single search result.
    The api_server.py expects these exact fields - do not change!
    """
    chunk_id: str
    paper_id: str
    title: str
    authors: str
    text: str
    score: float
    chunk_type: str = ""
    chunk_section: str = ""
    pdf_url: Optional[str] = None
    github_link: Optional[str] = None
    video_link: Optional[str] = None
    acm_url: Optional[str] = None
    abstract_url: Optional[str] = None


@dataclass
class RetrievalPipelineConfig:
    """Configuration for the retrieval pipeline."""
    qdrant_url: str
    qdrant_api_key: str
    openrouter_api_key: str
    embedding_model: str = "baai/bge-large-en-v1.5"
    chunks_path: str = "./chunks.json"
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    use_reranker: bool = True
    cohere_api_key: Optional[str] = None


class OpenRouterEmbedder:
    """
    Generate embeddings using OpenRouter API.
    Used to embed user queries for semantic search.
    """
    
    def __init__(self, api_key: str, model: str = "baai/bge-large-en-v1.5"):
        if not api_key:
            raise ValueError("An OpenRouter API key is required to generate query embeddings")
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
    
    def embed_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Cannot embed an empty query")

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        response_data = response.json()
        try:
            embedding = np.asarray(response_data["data"][0]["embedding"], dtype=np.float32)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("OpenRouter returned an invalid embedding response") from exc
        if embedding.ndim != 1 or embedding.size == 0 or not np.isfinite(embedding).all():
            raise ValueError("OpenRouter returned an invalid embedding vector")
        return embedding


class BM25Index:
    """
    BM25 index for keyword search.
    This runs entirely locally - no API calls needed!
    BM25 is good at finding exact keyword matches that semantic search might miss.
    """
    
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.chunk_id_to_idx = {
            str(chunk["chunk_id"]): index
            for index, chunk in enumerate(chunks)
            if "chunk_id" in chunk
        }
        self.tokenized_docs = [self._tokenize(chunk.get("text", "")) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_docs) if any(self.tokenized_docs) else None
    
    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())
    
    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        if top_k <= 0 or self.bm25 is None:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked_indices = sorted(
            range(len(scores)), key=lambda index: float(scores[index]), reverse=True
        )
        return [
            (index, float(scores[index]))
            for index in ranked_indices[:top_k]
            if scores[index] > 0
        ]


class RetrievalPipeline:
    """
    Main retrieval pipeline combining semantic search + BM25 + reranking.
    This is what api_server.py uses to find relevant chunks.
    """
    
    def __init__(self, config: Optional[RetrievalPipelineConfig] = None):
        if config is None:
            config = RetrievalPipelineConfig(
                qdrant_url=os.getenv("QDRANT_URL", ""),
                qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY", ""),
                embedding_model=os.getenv("EMBEDDING_MODEL", "baai/bge-large-en-v1.5"),
                cohere_api_key=os.getenv("COHERE_API_KEY") or None,
                chunks_path=os.getenv("CHUNKS_PATH") or "./chunks.json",
                use_reranker=os.getenv("USE_RERANKER", "true").lower()
                in {"1", "true", "yes", "on"},
            )

        missing = [
            name
            for name, value in (
                ("QDRANT_URL", config.qdrant_url),
                ("QDRANT_API_KEY", config.qdrant_api_key),
                ("OPENROUTER_API_KEY", config.openrouter_api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"Missing required environment/config values: {', '.join(missing)}"
            )

        chunks_path = Path(config.chunks_path).expanduser()
        if not chunks_path.is_absolute():
            chunks_path = Path(__file__).resolve().parent / chunks_path
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Chunks JSON file not found: {chunks_path}")

        with chunks_path.open("r", encoding="utf-8") as chunks_file:
            data = json.load(chunks_file)
        chunks = data.get("chunks") if isinstance(data, dict) else None
        if not isinstance(chunks, list):
            raise ValueError(f"Expected a 'chunks' list in {chunks_path}")
        for index, chunk in enumerate(chunks):
            if (
                not isinstance(chunk, dict)
                or not chunk.get("chunk_id")
                or not isinstance(chunk.get("text"), str)
            ):
                raise ValueError(
                    f"Chunk at index {index} must contain a chunk_id and string text"
                )

        self.config = config
        self.chunks = chunks
        self.qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
        self.embedder = OpenRouterEmbedder(
            api_key=config.openrouter_api_key, model=config.embedding_model
        )
        self.bm25_index = BM25Index(self.chunks)
        logger.info("BM25 index built from %d chunks", len(self.chunks))
    
    def semantic_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform semantic search using Qdrant.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of result dicts with chunk_id, score, and payload
        """
        if top_k <= 0:
            return []
        query_embedding = self.embedder.embed_query(query)
        points = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {
                "chunk_id": str(point.payload.get("chunk_id", point.id)),
                "score": float(point.score),
                "payload": point.payload,
            }
            for point in points
            if point.payload is not None
        ]
    
    def bm25_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform BM25 keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of result dicts with chunk_id, score, and payload
        """
        return [
            {
                "chunk_id": str(self.chunks[index]["chunk_id"]),
                "score": score,
                "payload": self.chunks[index],
            }
            for index, score in self.bm25_index.search(query, top_k)
        ]
    
    def hybrid_search(self, query: str, semantic_top_k: int = 30, bm25_top_k: int = 30) -> list[dict]:
        """
        Combine semantic and BM25 results using weighted scoring.
        
        Args:
            query: Search query
            semantic_top_k: Max results from semantic search
            bm25_top_k: Max results from BM25 search
            
        Returns:
            Combined and sorted list of results
        """
        semantic_results = self.semantic_search(query, semantic_top_k)
        bm25_results = self.bm25_search(query, bm25_top_k)

        def normalize(results: list[dict]) -> dict[str, float]:
            maximum = max((float(result["score"]) for result in results), default=0.0)
            if maximum <= 0:
                return {result["chunk_id"]: 0.0 for result in results}
            return {
                result["chunk_id"]: max(0.0, float(result["score"])) / maximum
                for result in results
            }

        semantic_scores = normalize(semantic_results)
        bm25_scores = normalize(bm25_results)
        combined: dict[str, dict] = {}
        for result in semantic_results:
            chunk_id = result["chunk_id"]
            combined[chunk_id] = {
                **result,
                "semantic_score": semantic_scores[chunk_id],
                "bm25_score": 0.0,
            }
        for result in bm25_results:
            chunk_id = result["chunk_id"]
            if chunk_id in combined:
                combined[chunk_id]["bm25_score"] = bm25_scores[chunk_id]
            else:
                combined[chunk_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "bm25_score": bm25_scores[chunk_id],
                }

        for result in combined.values():
            result["combined_score"] = (
                self.config.semantic_weight * result["semantic_score"]
                + self.config.bm25_weight * result["bm25_score"]
            )
        return sorted(
            combined.values(), key=lambda result: result["combined_score"], reverse=True
        )
    
    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """
        Rerank results using Cohere API (optional but improves quality).
        
        Args:
            query: Original query
            results: Results from hybrid_search
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list of results
        """
        if top_k <= 0 or not results:
            return []
        if not self.config.cohere_api_key:
            return results[:top_k]

        try:
            response = requests.post(
                "https://api.cohere.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.config.cohere_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "rerank-english-v3.0",
                    "query": query,
                    "documents": [result["payload"].get("text", "") for result in results],
                    "top_n": min(top_k, len(results)),
                },
                timeout=30,
            )
            response.raise_for_status()
            reranked = []
            for item in response.json()["results"]:
                index = int(item["index"])
                if not 0 <= index < len(results):
                    raise ValueError(f"Cohere returned invalid result index: {index}")
                result = dict(results[index])
                result["rerank_score"] = float(item["relevance_score"])
                reranked.append(result)
            return reranked[:top_k]
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning(
                "Cohere reranking failed; using hybrid ranking instead: %s", exc
            )
            return results[:top_k]
    
    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        use_reranker: Optional[bool] = None,
    ) -> list[RetrievalResult]:
        """
        Full retrieval pipeline - THIS IS WHAT api_server.py CALLS!
        
        Args:
            query: User's search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult objects ready for RAG generation
        """
        if top_k <= 0 or not query.strip():
            return []

        candidates = self.hybrid_search(query)
        should_rerank = (
            self.config.use_reranker if use_reranker is None else use_reranker
        )
        if should_rerank:
            candidates = self.rerank(
                query, candidates, top_k=min(top_k * 2, len(candidates))
            )

        results = []
        for result in candidates[:top_k]:
            payload = result["payload"]
            results.append(
                RetrievalResult(
                    chunk_id=str(payload.get("chunk_id", result.get("chunk_id", ""))),
                    paper_id=str(payload.get("paper_id", "")),
                    title=str(payload.get("title", "Untitled paper")),
                    authors=str(payload.get("authors", "")),
                    text=str(payload.get("text", "")),
                    score=float(
                        result.get(
                            "rerank_score",
                            result.get("combined_score", result.get("score", 0.0)),
                        )
                    ),
                    chunk_type=str(payload.get("chunk_type", "")),
                    chunk_section=str(payload.get("chunk_section", "")),
                    pdf_url=payload.get("pdf_url"),
                    github_link=payload.get("github_link"),
                    video_link=payload.get("video_link"),
                    acm_url=payload.get("acm_url"),
                    abstract_url=payload.get("abstract_url"),
                )
            )
        return results


# For testing this file directly
if __name__ == "__main__":
    import sys
    
    query = sys.argv[1] if len(sys.argv) > 1 else "3D Gaussian Splatting"
    
    print(f"Testing retrieval pipeline with query: '{query}'")
    print("=" * 60)
    
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve(query, top_k=5)
    
    print(f"\nFound {len(results)} results:\n")
    
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.title[:60]}...")
        print(f"   Paper ID: {r.paper_id}")
        print(f"   Text preview: {r.text[:100]}...")
        print()
