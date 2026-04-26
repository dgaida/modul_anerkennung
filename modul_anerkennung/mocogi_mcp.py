import os
import httpx
import numpy as np
from fastmcp import FastMCP
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Initialize FastMCP server
mcp = FastMCP("Mocogi API Server")

API_BASE_URL = "https://module.gm.th-koeln.de/api"

# Lazy loading of the embedding model
_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_headers():
    headers = {}
    token = os.getenv("MOCOGI_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@mcp.tool()
async def list_study_programs(filter: str = "currently-active") -> List[Dict[str, Any]]:
    """
    Gibt alle Studiengänge mit PO zurück.
    Filter-Optionen: 'currently-active' (default), 'not-expired', oder leer für alle.
    """
    async with httpx.AsyncClient() as client:
        params = {"filter": filter} if filter else {}
        response = await client.get(
            f"{API_BASE_URL}/studyPrograms", params=params, headers=get_headers()
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def get_modules_by_po(po_id: str) -> List[Dict[str, Any]]:
    """
    Gibt alle aktiven Module für eine bestimmte Prüfungsordnung (PO) zurück.
    Beispiele für po_id: 'inf_mi5' (MI Bachelor PO-5), 'inf_mim5' (MI Master PO-5).
    """
    async with httpx.AsyncClient() as client:
        params = {"select": "metadata", "active": "true", "po": po_id}
        response = await client.get(
            f"{API_BASE_URL}/modules", params=params, headers=get_headers()
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def get_all_active_modules() -> List[Dict[str, Any]]:
    """
    Gibt alle aktiven Module zurück.
    """
    async with httpx.AsyncClient() as client:
        params = {"select": "metadata", "active": "true"}
        response = await client.get(
            f"{API_BASE_URL}/modules", params=params, headers=get_headers()
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def search_modules(
    po_id: str, search_term: Optional[str] = None, max_ects: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Sucht nach Modulen in einem Studiengang (PO), die eine hohe Ähnlichkeit mit einem Suchbegriff
    haben und maximal eine gewisse Anzahl ECTS haben.

    Args:
        po_id: Die ID der Prüfungsordnung (z.B. 'inf_mi5').
        search_term: Suchbegriff für die semantische Ähnlichkeitssuche im Modultitel.
        max_ects: Maximale Anzahl an ECTS-Punkten.
    """
    async with httpx.AsyncClient() as client:
        params = {"select": "metadata", "active": "true", "po": po_id}
        response = await client.get(
            f"{API_BASE_URL}/modules", params=params, headers=get_headers()
        )
        response.raise_for_status()
        modules = response.json()

    # Filter by max ECTS
    if max_ects is not None:
        modules = [
            m for m in modules if m.get("metadata", {}).get("ects", 0) <= max_ects
        ]

    # Semantic search by search term
    if search_term:
        model = get_model()
        titles = [m.get("metadata", {}).get("title", "") for m in modules]
        if titles:
            # Normalize embeddings to make dot product equivalent to cosine similarity
            search_embedding = model.encode([search_term], normalize_embeddings=True)[0]
            title_embeddings = model.encode(titles, normalize_embeddings=True)

            # Dot product similarity
            similarities = np.dot(title_embeddings, search_embedding)

            for m, sim in zip(modules, similarities):
                m["similarity"] = float(sim)

            # Sort by similarity descending
            modules.sort(key=lambda x: x.get("similarity", 0), reverse=True)

    return modules


if __name__ == "__main__":
    mcp.run()
