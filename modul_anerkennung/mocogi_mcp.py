import os
import httpx
from fastmcp import FastMCP
from typing import List, Dict, Any

# Initialize FastMCP server
mcp = FastMCP("Mocogi API Server")

API_BASE_URL = "https://module.gm.th-koeln.de/api"

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
        response = await client.get(f"{API_BASE_URL}/studyPrograms", params=params, headers=get_headers())
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_modules_by_po(po_id: str) -> List[Dict[str, Any]]:
    """
    Gibt alle aktiven Module für eine bestimmte Prüfungsordnung (PO) zurück.
    Beispiele für po_id: 'inf_mi5' (MI Bachelor PO-5), 'inf_mim5' (MI Master PO-5).
    """
    async with httpx.AsyncClient() as client:
        params = {
            "select": "metadata",
            "active": "true",
            "po": po_id
        }
        response = await client.get(f"{API_BASE_URL}/modules", params=params, headers=get_headers())
        response.raise_for_status()
        return response.json()

@mcp.tool()
async def get_all_active_modules() -> List[Dict[str, Any]]:
    """
    Gibt alle aktiven Module zurück.
    """
    async with httpx.AsyncClient() as client:
        params = {
            "select": "metadata",
            "active": "true"
        }
        response = await client.get(f"{API_BASE_URL}/modules", params=params, headers=get_headers())
        response.raise_for_status()
        return response.json()

if __name__ == "__main__":
    mcp.run()
