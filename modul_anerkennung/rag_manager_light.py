"""Verwaltung von RAG-Prozessen mit LightRAG und Groq LLM."""

import os
import asyncio
from typing import Any
import torch
import numpy as np
from lightrag import LightRAG, QueryParam
# from lightrag.llm.openai import openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc
from transformers import AutoTokenizer, AutoModel
from lightrag.llm.hf import hf_embed
from openai import OpenAI
from .config import API_KEY, BASE_URL, RAG_STORAGE_DIR

# Logging aktivieren
setup_logger("lightrag", level="INFO")


# ----------------------------------------------------------
#  LLM- und Embedding-Funktionen
# ----------------------------------------------------------

async def llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict] = [],
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    """
    Asynchrone LLM-Abfrage über die Groq API.

    Args:
        prompt (str): Der Eingabetext oder die Frage.
        system_prompt (str | None): Optionaler Systemkontext.
        history_messages (list[dict]): Gesprächsverlauf für Kontext.
        keyword_extraction (bool): Flag für Schlüsselwortextraktion.
        **kwargs: Weitere Modellparameter.

    Returns:
        str: Antworttext des Modells.
    """
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    system_content = system_prompt or "You are an expert in module recognition and academic equivalence evaluation."

    # Groq-kompatible Promptstruktur
    messages = [
        {"role": "system", "content": system_content},
        *history_messages,
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.responses.create(
            model="openai/gpt-oss-20b",
            input=messages,
        )
        return response.output_text
    except Exception as e:
        return f"[Groq LLM Error] {e}"


async def embedding_func(texts: list[str]) -> np.ndarray:
    """
    Erstellt Text-Embeddings mithilfe eines Modells von Hugging Face.

    Args:
        texts (list[str]): Liste von Texten zur Einbettung.

    Returns:
        np.ndarray: Array von Embedding-Vektoren.
    """
    # Lade Tokenizer und Modell nur einmal (Performance)
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Stelle sicher, dass das Modell im Eval-Modus ist und GPU (falls verfügbar) nutzt
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Verwende hf_embed (bereitgestellt durch LightRAG)
    embeddings = await hf_embed(
        texts,
        tokenizer=tokenizer,
        embed_model=model,
        device=device
    )

    # Rückgabe als NumPy-Array für LightRAG-Kompatibilität
    return np.array(embeddings)


# ----------------------------------------------------------
#  LightRAG Manager-Klasse
# ----------------------------------------------------------

class LightRAGManager:
    """Kapselt die Verwendung von LightRAG mit Groq LLM."""

    def __init__(self) -> None:
        """
        Initialisiert den LightRAG-Manager.
        """
        self.rag: LightRAG | None = None

    async def initialize(self) -> None:
        """Initialisiert LightRAG mit Embeddings und LLM-Modell."""
        embedding_wrapper = EmbeddingFunc(
            embedding_dim=4096,
            func=embedding_func,
        )

        self.rag = LightRAG(
            working_dir=RAG_STORAGE_DIR,
            llm_model_func=llm_model_func,
            embedding_func=embedding_wrapper,
        )

        await self.rag.initialize_storages()
        await initialize_pipeline_status()

    async def insert_text(self, text: str) -> None:
        """
        Fügt einen neuen Text in die RAG-Datenbank ein.

        Args:
            text (str): Textinhalt, z. B. eine Modulbeschreibung.
        """
        if not self.rag:
            await self.initialize()
        await self.rag.ainsert(text)

    async def query(self, query_text: str, mode: str = "hybrid") -> Any:
        """
        Führt eine Abfrage in der RAG-Datenbank durch.

        Args:
            query_text (str): Suchanfrage.
            mode (str): Suchmodus (z. B. 'hybrid', 'semantic', 'keyword').

        Returns:
            Any: Antwortobjekt des RAG-Systems.
        """
        if not self.rag:
            await self.initialize()

        param = QueryParam(mode=mode)
        return await self.rag.aquery(query_text, param=param)

    async def finalize(self) -> None:
        """Beendet LightRAG und schließt alle Speicherschnittstellen."""
        if self.rag:
            await self.rag.finalize_storages()


# ----------------------------------------------------------
#  Beispielhafte Nutzung (nur für direkte Ausführung)
# ----------------------------------------------------------

if __name__ == "__main__":
    async def main():
        manager = LightRAGManager()
        await manager.initialize()

        await manager.insert_text("Grundlagen der Informatik: Einführung in Programmierung und Datenstrukturen.")
        result = await manager.query("Was sind die zentralen Themen des Moduls?")
        print(result)

        await manager.finalize()

    asyncio.run(main())
