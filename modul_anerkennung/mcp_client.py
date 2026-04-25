import sys
from fastmcp import Client
from fastmcp.client.transports.stdio import StdioTransport
from typing import List, Dict, Any

class MocogiClient:
    """Ein Client für den Mocogi MCP Server, der die Kommunikation über stdio ermöglicht.

    Dieser Client startet den Mocogi MCP Server als Subprozess und ermöglicht
    den Zugriff auf dessen Tools über eine asynchrone Schnittstelle.
    """
    def __init__(self):
        """Initialisiert den MocogiClient und konfiguriert den Server-Befehl."""
        # Der Server wird als Subprozess gestartet
        transport = StdioTransport(
            command=sys.executable,
            args=["-m", "modul_anerkennung.mocogi_mcp"]
        )
        self.client = Client(transport)

    async def __aenter__(self):
        """Ermöglicht die Nutzung des Clients als asynchroner Kontextmanager.

        Returns:
            MocogiClient: Die Instanz des Clients.
        """
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Beendet den Client beim Verlassen des asynchronen Kontextmanagers.

        Args:
            exc_type: Der Typ der Ausnahme, falls eine aufgetreten ist.
            exc_val: Der Wert der Ausnahme, falls eine aufgetreten ist.
            exc_tb: Der Traceback der Ausnahme, falls eine aufgetreten ist.
        """
        await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def list_study_programs(self, filter: str = "currently-active") -> List[Dict[str, Any]]:
        """Listet alle Studiengänge der TH Köln auf.

        Args:
            filter (str, optional): Filter für die Studiengänge (z.B. 'currently-active').
                Standardwert ist "currently-active".

        Returns:
            List[Dict[str, Any]]: Eine Liste von Studiengängen als Dictionaries.
        """
        result = await self.client.call_tool("list_study_programs", {"filter": filter})
        return result

    async def get_modules_by_po(self, po_id: str) -> List[Dict[str, Any]]:
        """Gibt alle aktiven Module für eine bestimmte Prüfungsordnung (PO) zurück.

        Args:
            po_id (str): Die ID der Prüfungsordnung (z.B. 'inf_mi5').

        Returns:
            List[Dict[str, Any]]: Eine Liste von Modulen als Dictionaries.
        """
        result = await self.client.call_tool("get_modules_by_po", {"po_id": po_id})
        return result

    async def get_all_active_modules(self) -> List[Dict[str, Any]]:
        """Gibt eine Liste aller aktiven Module zurück.

        Returns:
            List[Dict[str, Any]]: Eine Liste aller aktiven Module.
        """
        result = await self.client.call_tool("get_all_active_modules")
        return result

    async def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> Any:
        """Führt einen generischen Aufruf eines MCP-Tools aus.

        Args:
            name (str): Der Name des aufzurufenden Tools.
            arguments (Dict[str, Any], optional): Die Argumente für das Tool.
                Standardwert ist None.

        Returns:
            Any: Das Ergebnis des Tool-Aufrufs.
        """
        return await self.client.call_tool(name, arguments or {})

    async def list_tools(self) -> List[Any]:
        """Listet alle verfügbaren Tools des MCP Servers auf.

        Returns:
            List[Any]: Eine Liste der verfügbaren Tools.
        """
        return await self.client.list_tools()
