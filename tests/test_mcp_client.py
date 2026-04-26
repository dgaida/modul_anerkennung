import pytest
from modul_anerkennung.mcp_client import MocogiClient


@pytest.mark.asyncio
async def test_mcp_client_connection():
    mcp_client = MocogiClient()
    async with mcp_client as mcp:
        tools = await mcp.list_tools()
        assert len(tools) > 0
        names = [t.name for t in tools]
        assert "list_study_programs" in names
        assert "get_modules_by_po" in names
        assert "get_all_active_modules" in names
