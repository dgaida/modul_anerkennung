import pytest
from unittest.mock import AsyncMock, patch
from scripts.migrate_po_content import parse_markdown_table, migrate_content

def test_parse_markdown_table(tmp_path):
    md_content = """
| Modul in PO2 | Modul in PO3 |
|--------------|--------------|
| Algorithmik  | Algorithmen und Datenstrukturen |
| Mathe 1      | Grundlagen der Mathematik |
"""
    md_file = tmp_path / "test_mapping.md"
    md_file.write_text(md_content, encoding="utf-8")

    mappings = parse_markdown_table(str(md_file))
    assert len(mappings) == 2
    assert mappings[0] == ("Algorithmik", "Algorithmen und Datenstrukturen")
    assert mappings[1] == ("Mathe 1", "Grundlagen der Mathematik")

@pytest.mark.asyncio
async def test_migrate_content():
    mappings = [("Algorithmik", "Algorithmen und Datenstrukturen")]

    po2_modules = [
        {"module": {"id": "old_1", "metadata": {"title": "Algorithmik"}}}
    ]
    po3_modules = [
        {"module": {"id": "new_1", "metadata": {"title": "Algorithmen und Datenstrukturen"}}}
    ]

    full_source = {
        "id": "old_1",
        "metadata": {"title": "Algorithmik"},
        "deContent": {"content": "Alte Inhalte"},
        "enContent": {"content": "Old Content"}
    }

    full_target = {
        "id": "new_1",
        "metadata": {"title": "Algorithmen und Datenstrukturen"},
        "deContent": {},
        "enContent": {}
    }

    with patch("scripts.migrate_po_content.MocogiClient") as mock_client_class:
        mock_client = mock_client_class.return_value.__aenter__.return_value
        mock_client.get_modules_by_po.side_effect = [po2_modules, po3_modules]
        mock_client.get_module_details.side_effect = [full_source, full_target]
        mock_client.update_module = AsyncMock()

        await migrate_content("po2", "po3", mappings)

        # Verify update_module was called with correct data
        mock_client.update_module.assert_called_once()
        args, kwargs = mock_client.update_module.call_args
        target_id = args[0]
        updated_data = args[1]

        assert target_id == "new_1"
        assert updated_data["deContent"] == {"content": "Alte Inhalte"}
        assert updated_data["enContent"] == {"content": "Old Content"}
        assert updated_data["metadata"]["title"] == "Algorithmen und Datenstrukturen"
