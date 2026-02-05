"""Verwaltung von RAG-Prozessen und Dokumenten mit RAG-Anything."""
import asyncio
from typing import Any, List, Dict
import numpy as np
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc
from .config import RAG_STORAGE_DIR
from .llm_interface import LLMInterface


class RAGManager:
    """Kapselt die Verwendung von RAG-Anything."""

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        """Initialisiert den RAG-Manager."""
        self.llm_interface = LLMInterface(provider=provider, model=model)

        config = RAGAnythingConfig(
            working_dir=str(RAG_STORAGE_DIR),
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: List[Dict[str, str]] = [],
            **kwargs
        ) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            return await self.llm_interface.achat(messages, **kwargs)

        # Embedding function
        # llm_client currently doesn't provide a direct embedding function in a standard way across all providers
        # that easily plugs into LightRAG without knowing the dim.
        # So we might use a default one or try to use one from a provider.

        # If we have OpenAI, we can use their embedding.
        # For simplicity and robustness, we'll use a local one if possible or fallback.
        # LightRAG often uses OpenAI's text-embedding-3-large by default in examples.

        from lightrag.llm.openai import openai_embed
        import os

        async def embedding_func(texts: List[str]) -> np.ndarray:
            # Fallback to OpenAI if API key exists, else use a dummy or local one
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return await openai_embed(
                    texts,
                    model="text-embedding-3-large",
                    api_key=api_key
                )
            else:
                # Use a local model via sentence-transformers if available
                try:
                    from transformers import AutoTokenizer, AutoModel
                    import torch
                    from lightrag.llm.hf import hf_embed

                    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model.to(device)

                    return await hf_embed(texts, tokenizer, model, device)
                except ImportError:
                    # Very last fallback - random embeddings (for testing only)
                    print("WARNING: Using random embeddings!")
                    return np.random.rand(len(texts), 384)

        # Detect dimension for the embedding function
        # text-embedding-3-large is 3072, all-MiniLM-L6-v2 is 384
        emb_dim = 3072 if os.getenv("OPENAI_API_KEY") else 384

        self.embedding_wrapper = EmbeddingFunc(
            embedding_dim=emb_dim,
            max_token_size=8192,
            func=embedding_func,
        )

        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=llm_model_func, # Use same for vision for now
            embedding_func=self.embedding_wrapper,
        )

    async def process_document(self, file_path: str) -> None:
        """Indexiert ein Dokument in der RAG-Datenbank."""
        await self.rag.process_document_complete(file_path=file_path, parse_method="auto")

    async def query(self, query_text: str, mode: str = "hybrid") -> Any:
        """Führt eine Abfrage gegen die Wissensbasis aus."""
        return await self.rag.aquery(query_text, mode=mode)
