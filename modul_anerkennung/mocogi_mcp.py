"""MCP-Server-Implementierung für den Zugriff auf die Mocogi-API der TH Köln."""

import logging
import os
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

import modul_anerkennung.config  # noqa: F401 (ensure env is loaded)

logger = logging.getLogger(__name__)

API_BASE_URL = "https://module.gm.th-koeln.de/api"

# Initialize FastMCP server
mcp = FastMCP("Mocogi API Server")

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
        # Masked token for logging
        masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
        logger.debug(f"Using MOCOGI_API_TOKEN: {masked}")
        headers["Authorization"] = f"Bearer {token}"
    else:
        logger.warning("MOCOGI_API_TOKEN is NOT set!")
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
        logger.debug("Requesting module drafts from API...")
        response = await client.get(
            f"{API_BASE_URL}/moduleDrafts", headers=get_headers()
        )
        if response.status_code != 200:
            logger.error(f"Failed to get module drafts: {response.status_code} - {response.text}")
        response.raise_for_status()
        return response.json().get("direct", [])


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
            logger.info(f"Abfrage publizierter Module für PO {po_id}...")
            response = await client.get(
                f"{API_BASE_URL}/modules", params=params, headers=get_headers()
            )
            if response.status_code == 200:
                data = response.json()
                logger.info(f"  {len(data)} publizierte Module gefunden.")
                all_raw_items.extend(data)
            else:
                logger.warning(f"  Status {response.status_code} bei /modules: {response.text}")
        except Exception as e:
            logger.error(f"  Fehler bei /modules: {e}")

    # 2. Hole Entwürfe und filtere nach PO
    try:
        logger.info(f"Abfrage Modul-Entwürfe (Drafts) für PO {po_id}...")
        drafts = await get_module_drafts()
        logger.info(f"  Insgesamt {len(drafts)} Drafts gefunden.")
        found_drafts = 0
        for d in drafts:
            # Filtere nach PO in mandatoryPOs oder optionalPOs
            mandatory = d.get("mandatoryPOs", [])
            optional = d.get("optionalPOs", [])
            if po_id in mandatory or po_id in optional:
                logger.debug(f"  Draft: {d.get('module', {}).get('title')} | Mandatory: {mandatory} | Optional: {optional}")
                all_raw_items.append(d)
                found_drafts += 1
        logger.info(f"  {found_drafts} passende Drafts für {po_id} gefunden.")
    except Exception as e:
        logger.error(f"  Fehler bei get_module_drafts: {e}")

    # Standardisierung der Ergebnisse für den Service-Layer
    standardized_modules = []
    for item in all_raw_items:
        # Basis-Struktur
        standardized = item.copy()
        is_draft = "moduleDraftState" in item or item.get("isDraft", False)
        standardized["isDraft"] = is_draft

        # 1. Fall: /modules?select=metadata (flach)
        if "metadata" in item and isinstance(item["metadata"], dict) and "title" in item["metadata"]:
            standardized["metadata"] = item["metadata"]
            standardized["id"] = item.get("id") or item["metadata"].get("id")

        # 2. Fall: Gewrapped (publiziert oder Draft)
        elif "module" in item and isinstance(item["module"], dict):
            module_part = item["module"]
            if "metadata" in module_part:
                # Publiziert gewrapped
                standardized["metadata"] = module_part["metadata"]
                standardized["id"] = module_part.get("id") or module_part["metadata"].get("id")
            else:
                # Draft Struktur
                # WICHTIG: Für Drafts ist item["id"] die Draft-ID, die für Updates benötigt wird.
                # module_part["id"] ist die ID des zugrundeliegenden Moduls.
                draft_id = item.get("id")
                standardized["metadata"] = {
                    "title": module_part.get("title"),
                    "ects": item.get("ects") or module_part.get("ects"),
                    "abbreviation": module_part.get("abbreviation"),
                    "id": module_part.get("id")
                }
                standardized["id"] = draft_id or module_part.get("id")

        if standardized.get("metadata", {}).get("title"):
            standardized_modules.append(standardized)

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
