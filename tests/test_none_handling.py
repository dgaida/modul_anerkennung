import pytest
from modul_anerkennung.mocogi_mcp import get_modules_by_po
from unittest.mock import patch, AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_get_modules_by_po_handles_none_pos():
    """Testet, ob get_modules_by_po mit None-Werten in mandatoryPOs/optionalPOs umgehen kann."""

    # Mock für get_module_drafts
    mock_drafts = [
        {
            "id": "draft1",
            "moduleDraftState": "draft",
            "module": {"title": "Test Modul 1", "id": "m1"},
            "mandatoryPOs": None,  # Hier ist der None-Wert
            "optionalPOs": ["inf_inf3"]
        },
        {
            "id": "draft2",
            "moduleDraftState": "draft",
            "module": {"title": "Test Modul 2", "id": "m2"},
            "mandatoryPOs": ["inf_inf3"],
            "optionalPOs": None   # Hier ist der None-Wert
        }
    ]

    with patch("modul_anerkennung.mocogi_mcp.get_module_drafts", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_drafts

        # Mock für httpx.AsyncClient.get
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_http_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_http_get.return_value = mock_response

            # Ausführung
            result = await get_modules_by_po("inf_inf3")

            # Verifikation
            assert len(result) == 2
            assert result[0]["metadata"]["title"] == "Test Modul 1"
            assert result[1]["metadata"]["title"] == "Test Modul 2"

@pytest.mark.asyncio
async def test_map_to_protocol_update_handles_none():
    """Testet die map_to_protocol_update Funktion in migrate_po_content.py."""
    from scripts.migrate_po_content import map_to_protocol_update

    # Test-Daten mit None-Werten
    full_data = {
        "module": {
            "title": "Test",
            "examPhases": None,
            "assessmentMethods": None
        },
        "mandatoryPOs": None,
        "optionalPOs": None
    }

    # Darf keine Exception werfen
    result = map_to_protocol_update(full_data)

    assert result["metadata"]["examPhases"] == []
    assert result["metadata"]["assessmentMethods"] == {}
    assert result["metadata"]["po"] == []
