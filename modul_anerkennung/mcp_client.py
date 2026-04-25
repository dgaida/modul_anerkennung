import sys
from fastmcp import Client
from typing import List, Dict, Any

class MocogiClient:
    """
    Ein Client für den Mocogi MCP Server, der die Kommunikation über stdio ermöglicht.
    """
    def __init__(self):
        # Der Server wird als Subprozess gestartet
        cmd = f"{sys.executable} -m modul_anerkennung.mocogi_mcp"
        self.client = Client(cmd)

    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def list_study_programs(self, filter: str = "currently-active") -> List[Dict[str, Any]]:
        """Listet alle Studiengänge auf."""
        result = await self.client.call_tool("list_study_programs", {"filter": filter})
        return result

    async def get_modules_by_po(self, po_id: str) -> List[Dict[str, Any]]:
        """Gibt Module einer bestimmten Prüfungsordnung zurück."""
        result = await self.client.call_tool("get_modules_by_po", {"po_id": po_id})
        return result

    async def get_all_active_modules(self) -> List[Dict[str, Any]]:
        """Gibt alle aktiven Module zurück."""
        result = await self.client.call_tool("get_all_active_modules")
        return result

    async def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> Any:
        """Generischer Aufruf eines Tools."""
        return await self.client.call_tool(name, arguments or {})
