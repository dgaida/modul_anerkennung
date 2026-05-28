import pytest

from unittest.mock import patch, MagicMock, AsyncMock
from modul_anerkennung.mocogi_mcp import get_modules_by_po, search_modules

@pytest.mark.asyncio
async def test_get_modules_by_po_merges_drafts():
    po_id = "inf_inf3"

    # Mock for published modules (returns empty for inf_inf3)
    published_modules = []

    # Mock for drafts
    drafts = [
        {
            "ects": 6,
            "module": {
                "id": "8cff2c5b-6f2f-4d8f-8101-f74f30c0a603",
                "title": "Algorithmen und Datenstrukturen",
                "abbreviation": "Algo"
            },
            "mandatoryPOs": ["inf_inf3"],
            "optionalPOs": [],
            "moduleDraftState": "published"
        },
        {
            "ects": 5,
            "module": {
                "id": "other-id",
                "title": "Other Module",
                "abbreviation": "OM"
            },
            "mandatoryPOs": ["other_po"],
            "optionalPOs": [],
            "moduleDraftState": "draft"
        }
    ]

    with patch("httpx.AsyncClient.get") as mock_get:
        # Mock responses
        res_published = MagicMock()
        res_published.status_code = 200
        res_published.json.return_value = published_modules

        res_drafts = MagicMock()
        res_drafts.status_code = 200
        res_drafts.json.return_value = {"direct": drafts}

        mock_get.side_effect = [res_published, res_drafts]

        results = await get_modules_by_po(po_id)

        # Should only contain the one draft for inf_inf3
        assert len(results) == 1
        assert results[0]["metadata"]["title"] == "Algorithmen und Datenstrukturen"
        assert results[0]["metadata"]["ects"] == 6
        assert results[0].get("isDraft") is True

@pytest.mark.asyncio
async def test_search_modules_with_drafts():
    po_id = "inf_inf3"

    # Mock for get_modules_by_po to return a draft
    mock_modules = [
        {
            "metadata": {
                "title": "Algorithmen und Datenstrukturen",
                "ects": 6,
                "id": "8cff2c5b-6f2f-4d8f-8101-f74f30c0a603"
            },
            "isDraft": True
        }
    ]

    with patch("modul_anerkennung.mocogi_mcp.get_modules_by_po", new_callable=AsyncMock) as mock_get_po:
        mock_get_po.return_value = mock_modules

        # Search for "Algorithmen"
        results = await search_modules(po_id, search_term="Algorithmen")

        assert len(results) == 1
        assert results[0]["metadata"]["title"] == "Algorithmen und Datenstrukturen"
        assert "similarity" in results[0]
        assert results[0]["similarity"] > 0.5
