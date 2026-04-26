import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from modul_anerkennung.services import RecognitionService
from modul_anerkennung.models import ModuleAnalysis, ComparisonReport


@pytest.mark.asyncio
async def test_analyze_module():
    # Mock LLMInterface
    mock_llm = MagicMock()
    mock_llm.achat = AsyncMock(
        return_value='{"name": "Test", "ects": 5, "keywords": ["a", "b"]}'
    )
    mock_llm.extract_json.return_value = ModuleAnalysis(
        name="Test", ects=5, keywords=["a", "b"]
    )

    service = RecognitionService(llm=mock_llm)
    result = await service.analyze_module("Some text")

    assert result.name == "Test"
    assert result.ects == 5
    assert len(result.keywords) == 2
    mock_llm.achat.assert_called_once()


@pytest.mark.asyncio
async def test_perform_comparison():
    mock_llm = MagicMock()
    mock_llm.achat = AsyncMock(return_value="{}")
    mock_llm.extract_json.return_value = ComparisonReport(
        decision="Ja", reasoning="Good", report="Full report"
    )

    service = RecognitionService(llm=mock_llm)
    result = await service.perform_comparison("Ext", {"metadata": {"title": "Int"}})

    assert result.decision == "Ja"
    assert result.reasoning == "Good"
    mock_llm.achat.assert_called_once()


@pytest.mark.asyncio
async def test_search_and_compare():
    mock_llm = MagicMock()
    # Mock perform_comparison to avoid nested calls logic
    mock_report = ComparisonReport(decision="Ja", reasoning="Ok", report="R")

    service = RecognitionService(llm=mock_llm)
    service.perform_comparison = AsyncMock(return_value=mock_report)

    # Mock MocogiClient
    with patch("modul_anerkennung.services.MocogiClient") as mock_client_class:
        mock_client_instance = mock_client_class.return_value.__aenter__.return_value
        mock_client_instance.call_tool = AsyncMock(
            return_value=[{"metadata": {"title": "M1"}}]
        )

        results = await service.search_and_compare("po1", "key", "5", "ext text")

        assert len(results) == 1
        assert results[0][0]["metadata"]["title"] == "M1"
        assert results[0][1].decision == "Ja"
