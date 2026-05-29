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
    Halt alle Modulentwürfe ab, filtert nach inf_inf3 Pflichtmodulen und gibt sie aus.
    Anschließend werden Details für "Algorithmen und Datenstrukturen" angezeigt.
    """
    logger.info("Starte Abfrage der inf_inf3 Pflichtmodule...")

    async with MocogiClient() as client:
        try:
            # Hole alle Entwürfe (wichtig für inf_inf3, da get_modules_by_po hier eventuell unvollständig ist)
            drafts = await client.get_module_drafts()
            logger.info(f"Insgesamt {len(drafts)} Entwürfe geladen.")

            inf3_mandatory = []
            target_matches = []
            target_title = "Algorithmen und Datenstrukturen"
            po_id = "inf_inf3"

            for draft in drafts:
                print(draft)
                # Prüfe mandatoryPOs
                mandatory = draft.get("mandatoryPOs") or []
                module_info = draft.get("module") or {}
                title = module_info.get("title", "Unbekanntes Modul")

                if po_id in mandatory:
                    inf3_mandatory.append(title)
                    if title == target_title:
                        target_matches.append(draft)

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
                    # Bei Drafts ist die Top-Level ID die Draft-ID, die für Detailabfragen benötigt wird
                    print(match)
                    draft_id = match.get("id")
                    if not draft_id:
                        # Fallback falls ID fehlt
                        module_info = match.get("module") or {}
                        draft_id = module_info.get("id")

                    if not draft_id:
                        logger.warning(f"Konnte keine ID für Modul '{target_title}' finden.")
                        continue

                    details = await client.get_module_draft_details(draft_id)

                    if len(target_matches) > 1:
                        print(f"\n--- Treffer {i+1} (ID: {draft_id}) ---")

                    # 1. Rohdaten
                    print("\n[Rohdaten]")
                    print(json.dumps(details, indent=2, ensure_ascii=False))

                    # 2. Schöner formatiert
                    print("\n[Formatiert]")
                    module = details.get("module", {})
                    print(f"Titel:          {module.get('title')}")
                    print(f"Kürzel:         {module.get('abbreviation')}")
                    print(f"ECTS:           {details.get('ects')}")

                    de_content = details.get("deContent") or {}
                    print("\nInhalt (DE):")
                    print(de_content.get("content", "Kein Inhalt vorhanden."))

                    print("\nLernergebnisse (DE):")
                    print(de_content.get("learningOutcomes", "Keine Lernergebnisse vorhanden."))

                    print("\nPrüfungsform:")
                    exams = details.get("examPhases") or []
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
