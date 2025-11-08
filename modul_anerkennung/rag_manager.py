"""Verwaltung von RAG-Prozessen und Dokumenten mit RAG-Anything."""
import asyncio
from typing import Any
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from .config import API_KEY, BASE_URL, RAG_STORAGE_DIR

class RAGManager:
    """Kapselt die Verwendung von RAG-Anything."""

    def __init__(self) -> None:
        """Initialisiert den RAG-Manager."""
        config = RAGAnythingConfig(
            working_dir=str(RAG_STORAGE_DIR),
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        def llm_model_func(prompt, **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                api_key=API_KEY,
                base_url=BASE_URL,
                **kwargs,
            )

        embedding_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=API_KEY,
                base_url=BASE_URL,
            ),
        )

        self.rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=llm_model_func,
            embedding_func=embedding_func,
        )

    async def process_document(self, file_path: str) -> None:
        """Indexiert ein Dokument in der RAG-Datenbank."""
        await self.rag.process_document_complete(file_path=file_path, parse_method="auto")

    async def query(self, query_text: str) -> Any:
        """Führt eine Abfrage gegen die Wissensbasis aus."""
        return await self.rag.aquery(query_text, mode="hybrid")
