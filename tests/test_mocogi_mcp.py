import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from modul_anerkennung.mocogi_mcp import search_modules

@pytest.mark.asyncio
async def test_search_modules_filtering():
    mock_modules = [
        {"metadata": {"title": "Math 1", "ects": 5}},
        {"metadata": {"title": "Math 2", "ects": 10}},
    ]

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_modules
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Test ECTS filter
        results = await search_modules("po123", max_ects=6)
        assert len(results) == 1
        assert results[0]["metadata"]["ects"] == 5

@pytest.mark.asyncio
async def test_search_modules_similarity():
    mock_modules = [
        {"metadata": {"title": "Mathematik für Informatiker", "ects": 5}},
        {"metadata": {"title": "Datenbanken", "ects": 5}},
    ]

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_modules
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Test similarity sorting
        results = await search_modules("po123", search_term="Mathe")
        assert len(results) == 2
        assert "similarity" in results[0]
        assert results[0]["metadata"]["title"] == "Mathematik für Informatiker"
        assert results[0]["similarity"] > results[1]["similarity"]
