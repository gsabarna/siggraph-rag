#!/usr/bin/env python3
"""
RAG Generation Pipeline for SIGGRAPH 2025 Papers.

Uses the retrieval pipeline to find relevant chunks,
then generates an answer using an LLM via OpenRouter API.

Usage:
    from rag_generate import RAGGenerator, GenerationConfig, SYSTEM_PROMPT
    
    generator = RAGGenerator()
    result = generator.generate("What is 3D Gaussian Splatting?")
    print(result["answer"])
"""

import os
import logging
import requests
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from retrieval_pipeline import RetrievalPipeline, RetrievalResult

logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPT - This tells the LLM how to behave
# =============================================================================
SYSTEM_PROMPT = """You are an expert research assistant specializing in computer graphics, specifically SIGGRAPH 2025 papers.

Your task is to answer questions using ONLY the provided research paper excerpts.

Rules:
1. Cite sources using [Paper Title] format
2. Be comprehensive and technically accurate
3. If the excerpts don't contain the answer, say so
4. Use LaTeX for math: $inline$ or $$block$$
5. Do NOT make up information not in the excerpts
6. Do NOT include a References section at the end
"""


# =============================================================================
# QUERY REFINEMENT PROMPT
# =============================================================================
QUERY_REFINEMENT_PROMPT = """You are an expert at refining search queries for academic paper retrieval.

Given a user's question, rewrite it as a clear, focused search query that will retrieve the most relevant research papers.

Keep it concise (under 20 words). Focus on key technical terms.

User question: {query}

Refined search query:"""


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class GenerationConfig:
    """Configuration for the RAG generator."""
    llm_provider: str = "openai"
    use_reranker: bool = True
    llm_model: str = "openai/gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2000
    openrouter_api_key: Optional[str] = None  # Will load from env if not set
    refine_query: bool = True  # Whether to refine queries before retrieval
    refinement_model: str = "openai/gpt-3.5-turbo"  # Cheaper model for refinement
    retrieval_top_k: int = 8  # Number of chunks to retrieve


# =============================================================================
# RAG GENERATOR CLASS
# =============================================================================
class RAGGenerator:
    """
    Main RAG class - this is what api_server.py uses!
    
    Flow:
    1. Refine the user's query (optional)
    2. Retrieve relevant chunks using the retrieval pipeline
    3. Format chunks into context
    4. Generate answer using LLM
    5. Return answer with source metadata
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None, retrieval_pipeline=None):
        if config is None:
            model = os.getenv("LLM_MODEL", "openai/gpt-4o").strip()
            if model and "/" not in model:
                model = f"openai/{model}"
            config = GenerationConfig(
                llm_model=model,
                temperature=float(os.getenv("TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("MAX_TOKENS", "2000")),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                refine_query=os.getenv("REFINE_QUERY", "true").lower()
                in {"1", "true", "yes", "on"},
                retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "8")),
            )
        self.config = config
        self.retrieval = retrieval_pipeline or RetrievalPipeline()
        self.openrouter_api_key = (
            self.config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        )
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for answer generation")
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
    
    def refine_query(self, query: str) -> str:
        if not self.config.refine_query or not query.strip():
            return query

        payload = {
            "model": self.config.refinement_model,
            "messages": [
                {
                    "role": "user",
                    "content": QUERY_REFINEMENT_PROMPT.format(query=query),
                }
            ],
            "temperature": 0.3,
            "max_tokens": 100,
        }
        try:
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            refined = response.json()["choices"][0]["message"]["content"].strip()
            refined = refined.strip("\"' ")
            return refined or query
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            logger.warning("Query refinement failed; using original query: %s", exc)
            return query
    
    def _format_context(self, results: list[RetrievalResult]) -> str:
        sources = []
        for index, result in enumerate(results, start=1):
            sources.append(
                f"--- Source {index} ---\n"
                f"Title: {result.title}\n"
                f"Authors: {result.authors}\n"
                f"Section: {result.chunk_section}\n\n"
                f"Content:\n{result.text}"
            )
        return "\n\n".join(sources)
    
    def _build_sources_metadata(self, results: list[RetrievalResult]) -> list[dict]:
        seen = {}
        for result in results:
            key = result.title.strip().casefold()
            if key and key not in seen:
                seen[key] = {
                    "title": result.title,
                    "authors": result.authors,
                    "pdf_url": result.pdf_url,
                    "github_link": result.github_link,
                    "video_link": result.video_link,
                    "acm_url": result.acm_url,
                    "abstract_url": result.abstract_url,
                }
        return list(seen.values())
    
    def _call_llm(self, query: str, context: str) -> str:
        user_message = (
            "Based on the following research paper excerpts, answer this question.\n\n"
            f"Question: {query}\n\n"
            f"Research Paper Excerpts:\n{context}\n\n"
            "Cite sources using [Paper Title] format."
        )
        response = requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.llm_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            },
            timeout=120,
        )
        print(f"LLM response status: {response.status_code}")
        print(f"LLM response content: {response.text}")
        response.raise_for_status()
        try:
            answer = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("OpenRouter returned an invalid chat completion response") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("OpenRouter returned an empty answer")
        return answer.strip()

    def generate(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        refine_query: Optional[bool] = None,
        use_reranker: Optional[bool] = None,
    ) -> dict:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        refined = query
        if refine_query is not False:
            refined = self.refine_query(query)
        results = self.retrieval.retrieve(
            refined,
            top_k=top_k if top_k is not None else self.config.retrieval_top_k,
            use_reranker=use_reranker,
        )
        if not results:
            return {
                "query": query,
                "refined_query": refined if refined != query else None,
                "answer": "I couldn't find any relevant papers to answer this question.",
                "sources": [],
            }

        answer = self._call_llm(query, self._format_context(results))
        return {
            "query": query,
            "refined_query": refined if refined != query else None,
            "answer": answer,
            "sources": self._build_sources_metadata(results) if return_sources else [],
        }


# =============================================================================
# CLI FOR TESTING
# =============================================================================
if __name__ == "__main__":
    import sys
    
    query = sys.argv[1] if len(sys.argv) > 1 else "What is 3D Gaussian Splatting?"
    
    print("Initializing RAG Generator...")
    generator = RAGGenerator()
    
    print(f"\nQuery: {query}")
    print("=" * 60)
    
    result = generator.generate(query)
    
    print(f"Refined Query: {result.get('refined_query', 'N/A')}")
    print("=" * 60)
    print("\nAnswer:")
    print(result['answer'])
    print("=" * 60)
    print(f"\nSources: {len(result.get('sources', []))} papers")
    for source in result.get('sources', []):
        print(f"  - {source['title']}")
