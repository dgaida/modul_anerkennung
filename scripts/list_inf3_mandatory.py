"""
Skript zum Auflisten aller Pflichtveranstaltungen der Prüfungsordnung inf_inf3.
Nutzt die Mocogi-API über den MCP-Client.
Handhabt publizierte Module und fehlende Details robust.
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
            # Hole alle Entwürfe
            try:
                drafts = await client.get_module_drafts()
                logger.info(f"Insgesamt {len(drafts)} Entwürfe geladen.")
            except Exception as e:
                logger.error(f"Konnte Entwürfe nicht laden: {e}")
                return

            inf3_mandatory = []
            target_matches = []
            target_title = "Algorithmen und Datenstrukturen"
            po_id = "inf_inf3"

            for draft in drafts:
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
                # Doppelte Einträge (z.B. durch direct/indirect) filtern
                unique_titles = sorted(list(set(inf3_mandatory)))
                print(f"\nPflichtveranstaltungen in {po_id} ({len(unique_titles)}):")
                print("-" * 50)
                for title in unique_titles:
                    print(f"- {title}")
                print("-" * 50)

            # Details für das gewünschte Modul ausgeben
            if target_matches:
                print(f"\nDetails für \"{target_title}\":")
                # Nur den ersten eindeutigen Treffer anzeigen um Redundanz zu vermeiden
                seen_ids = set()
                for match in target_matches:
                    draft_id = match.get("id")
                    module_info_summary = match.get("module") or {}
                    module_id = module_info_summary.get("id")

                    actual_id = draft_id or module_id
                    if actual_id in seen_ids:
                        continue
                    seen_ids.add(actual_id)

                    draft_state = match.get("moduleDraftState")

                    details = None
                    try:
                        # Falls publiziert, direkt die Modul-Details laden
                        if draft_state == "published" and module_id:
                            logger.debug(f"Modul ist publiziert (ID: {module_id}). Nutze get_module_details.")
                            details = await client.get_module_details(module_id)
                        elif draft_id:
                            logger.debug(f"Lade Draft-Details für ID: {draft_id}")
                            details = await client.get_module_draft_details(draft_id)
                    except Exception as e:
                        logger.warning(f"Konnte Details nicht von API laden ({e}).")

                    # Fallback-Struktur aufbauen, falls API-Details fehlen oder unvollständig sind (z.B. 500 error)
                    if not details or details.get("id") == "published":
                        logger.info("Nutze Zusammenfassungsdaten als Fallback für die Anzeige.")
                        details = {
                            "ects": match.get("ects"),
                            "module": module_info_summary,
                            "examPhases": match.get("examPhases") or []
                        }

                    # 1. Rohdaten
                    print("\n[Rohdaten]")
                    print(json.dumps(details, indent=2, ensure_ascii=False))

                    # 2. Schöner formatiert
                    print("\n[Formatiert]")
                    module = details.get("module", {})
                    metadata = details.get("metadata") or {}

                    res_title = module.get("title") or metadata.get("title") or module_info_summary.get("title")
                    res_abbrev = module.get("abbreviation") or metadata.get("abbrev") or metadata.get("abbreviation") or module_info_summary.get("abbreviation")
                    res_ects = details.get("ects") or metadata.get("ects") or match.get("ects")

                    print(f"Titel:          {res_title}")
                    print(f"Kürzel:         {res_abbrev}")
                    print(f"ECTS:           {res_ects}")

                    de_content = details.get("deContent") or {}
                    print("\nInhalt (DE):")
                    print(de_content.get("content", "Kein Inhalt vorhanden."))

                    print("\nLernergebnisse (DE):")
                    # Verschiedene mögliche Keys für Lernergebnisse prüfen
                    outcomes = de_content.get("learningOutcome") or de_content.get("learningOutcomes") or "Keine Lernergebnisse vorhanden."
                    print(outcomes)

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
            logger.error(f"Unerwarteter Fehler: {e}")

if __name__ == "__main__":
    asyncio.run(list_inf3_mandatory())
