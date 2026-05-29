"""
Skript zum Auflisten aller Pflichtveranstaltungen der Prüfungsordnung inf_inf3.
Nutzt die Mocogi-API über den MCP-Client.
"""

import asyncio
import json
import logging
from modul_anerkennung.mcp_client import MocogiClient

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("inf3_list")

async def list_inf3_mandatory():
    """
    Halt alle Module für inf_inf3 ab, filtert nach Pflichtmodulen und gibt sie aus.
    Anschließend werden Details für "Algorithmen und Datenstrukturen" angezeigt.
    """
    po_id = "inf_inf3"
    target_title = "Algorithmen und Datenstrukturen"
    logger.info(f"Starte Abfrage der {po_id} Pflichtmodule...")

    async with MocogiClient() as client:
        try:
            # Hole alle Module für die PO (standardisiert durch den MCP Server)
            modules = await client.get_modules_by_po(po_id)
            logger.info(f"Insgesamt {len(modules)} Module für {po_id} geladen.")

            inf3_mandatory = []
            target_matches = []

            for mod in modules:
                # get_modules_by_po liefert bereits gefilterte Module für die PO
                # Wir müssen nur prüfen, ob es ein Pflichtmodul ist
                mandatory_pos = mod.get("mandatoryPOs") or []
                metadata = mod.get("metadata") or {}
                title = metadata.get("title", "Unbekanntes Modul")

                if po_id in mandatory_pos:
                    inf3_mandatory.append(title)
                    if title == target_title:
                        target_matches.append(mod)

            if not inf3_mandatory:
                print(f"\nKeine Pflichtmodule für {po_id} gefunden.")
            else:
                print(f"\nPflichtveranstaltungen in {po_id} ({len(inf3_mandatory)}):")
                print("-" * 50)
                for title in sorted(inf3_mandatory):
                    print(f"- {title}")
                print("-" * 50)

            # Details für das gewünschte Modul ausgeben
            if target_matches:
                print(f"\nDetails für \"{target_title}\":")
                for i, match in enumerate(target_matches):
                    module_id = match.get("id")
                    is_draft = match.get("isDraft", False)

                    if not module_id:
                        logger.warning(f"Modul '{target_title}' hat keine ID.")
                        continue

                    if is_draft:
                        details = await client.get_module_draft_details(module_id)
                    else:
                        details = await client.get_module_details(module_id)

                    if len(target_matches) > 1:
                        print(f"\n--- Treffer {i+1} (ID: {module_id}, Draft: {is_draft}) ---")

                    # 1. Rohdaten
                    print("\n[Rohdaten]")
                    print(json.dumps(details, indent=2, ensure_ascii=False))

                    # 2. Schöner formatiert
                    print("\n[Formatiert]")

                    # Details-Struktur kann je nach Draft/Publiziert variieren
                    # Wir versuchen die wichtigsten Felder zu finden
                    module_data = details.get("module") if is_draft else details
                    if not module_data:
                        module_data = details

                    meta = details.get("metadata") if not is_draft else {}

                    title = module_data.get("title") or meta.get("title")
                    abbrev = module_data.get("abbreviation") or meta.get("abbreviation")
                    ects = details.get("ects") or meta.get("ects") or module_data.get("ects")

                    print(f"Titel:          {title}")
                    print(f"Kürzel:         {abbrev}")
                    print(f"ECTS:           {ects}")

                    de_content = details.get("deContent") or module_data.get("deContent") or {}
                    print("\nInhalt (DE):")
                    print(de_content.get("content", "Kein Inhalt vorhanden."))

                    print("\nLernergebnisse (DE):")
                    print(de_content.get("learningOutcomes", "Keine Lernergebnisse vorhanden."))

                    print("\nPrüfungsform:")
                    # Prüfungsformen können in metadata oder direkt liegen
                    exams = details.get("examPhases") or (meta.get("examPhases") if meta else [])
                    if not exams and module_data:
                        exams = module_data.get("examPhases") or []

                    if exams:
                        for exam in exams:
                            print(f"- {exam}")
                    else:
                        print("Keine Prüfungsformen angegeben.")

                    print("-" * 50)
            else:
                print(f"\nKein Modul mit dem Titel \"{target_title}\" in {po_id} gefunden.")

        except Exception as e:
            logger.error(f"Fehler bei der Abfrage: {e}")
            if "Authorization" in str(e) or "401" in str(e):
                logger.error("Bitte überprüfe den MOCOGI_API_TOKEN in deiner secrets.env.")

if __name__ == "__main__":
    asyncio.run(list_inf3_mandatory())
