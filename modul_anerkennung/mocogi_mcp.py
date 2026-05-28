"""MCP-Server-Implementierung für den Zugriff auf die Mocogi-API der TH Köln."""

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
    """Lädt das Embedding-Modell verzögert (Lazy Loading)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_headers(extra_headers: Optional[Dict[str, str]] = None):
    """Erstellt die HTTP-Header für die API-Anfragen, inkl. Bearer Token."""
    headers = extra_headers.copy() if extra_headers else {}
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
async def get_module_drafts() -> List[Dict[str, Any]]:
    """
    Gibt alle Modul-Entwürfe (Drafts) zurück, auf die der aktuelle Benutzer Zugriff hat.
    Entspricht "Meine Module" im Mocogi-Frontend.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/moduleDrafts", headers=get_headers()
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def get_modules_by_po(po_id: str) -> List[Dict[str, Any]]:
    """
    Gibt alle Module für eine bestimmte Prüfungsordnung (PO) zurück.
    Berücksichtigt sowohl publizierte Module als auch Entwürfe (Drafts).
    """
    all_raw_items = []

    # 1. Hole publizierte Module
    async with httpx.AsyncClient() as client:
        params = {"select": "metadata", "active": "true", "po": po_id}
        try:
            response = await client.get(
                f"{API_BASE_URL}/modules", params=params, headers=get_headers()
            )
            if response.status_code == 200:
                all_raw_items.extend(response.json())
        except Exception:
            # Wenn 404 o.ä., ignorieren wir das hier und schauen bei den Drafts
            pass

    # 2. Hole Entwürfe und filtere nach PO
    try:
        drafts = await get_module_drafts()
        for d in drafts:
            # Filtere nach PO in mandatoryPOs oder optionalPOs
            mandatory = d.get("mandatoryPOs", [])
            optional = d.get("optionalPOs", [])
            if po_id in mandatory or po_id in optional:
                all_raw_items.append(d)
    except Exception:
        pass

    # Standardisierung der Ergebnisse für den Service-Layer
    standardized_modules = []
    for item in all_raw_items:
        # Falls schon standardmäßig (durch /modules?select=metadata)
        if "metadata" in item and isinstance(item["metadata"], dict) and "title" in item["metadata"]:
            standardized_modules.append(item)
            continue

        # Falls gewrapped (durch /modules ohne select=metadata)
        if "module" in item and isinstance(item["module"], dict) and "metadata" in item["module"]:
            new_item = item.copy()
            new_item["metadata"] = item["module"]["metadata"]
            standardized_modules.append(new_item)
            continue

        # Falls Draft-Struktur
        module_part = item.get("module", {})
        if isinstance(module_part, dict):
            metadata = {
                "title": module_part.get("title"),
                "ects": item.get("ects") or module_part.get("ects"),
                "abbreviation": module_part.get("abbreviation"),
                "id": module_part.get("id")
            }
            new_item = item.copy()
            new_item["metadata"] = metadata
            new_item["isDraft"] = True
            standardized_modules.append(new_item)

    return standardized_modules


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
    """
    modules = await get_modules_by_po(po_id)

    # Filter by max ECTS
    if max_ects is not None:
        modules = [
            m for m in modules if m.get("metadata", {}).get("ects") is not None
            and m.get("metadata", {}).get("ects", 0) <= max_ects
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


@mcp.tool()
async def get_module_details(module_id: str) -> Dict[str, Any]:
    """
    Gibt die vollständigen Details eines Moduls zurück.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/modules/{module_id}", headers=get_headers()
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def get_module_draft_details(module_id: str) -> Dict[str, Any]:
    """
    Gibt die vollständigen Details eines Modul-Entwurfs (Draft) zurück.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/moduleDrafts/{module_id}", headers=get_headers()
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def update_module(module_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aktualisiert ein publiziertes Modul mit den übergebenen Daten (PUT).
    """
    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{API_BASE_URL}/modules/{module_id}", json=data, headers=get_headers()
        )
        response.raise_for_status()
        return response.json()


@mcp.tool()
async def update_module_draft(module_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aktualisiert einen Modul-Entwurf (Draft) mit den übergebenen Daten (PUT).
    Erfordert das Mocogi-Version-Scheme v1.0s.
    """
    headers = get_headers({"Content-Type": "application/json", "Mocogi-Version-Scheme": "v1.0s"})
    async with httpx.AsyncClient() as client:
        response = await client.put(
            f"{API_BASE_URL}/moduleDrafts/{module_id}",
            json=data,
            headers=headers
        )
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    mcp.run()
