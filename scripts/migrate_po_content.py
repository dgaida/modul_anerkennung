#!/usr/bin/env python3
"""
Skript zur Migration von Modulinhalten zwischen zwei Prüfungsordnungen (PO).
Basierend auf einer Äquivalenzliste in einer Markdown-Datei.
"""

import asyncio
import argparse
import logging
from typing import List, Tuple, Dict, Any
from modul_anerkennung.mcp_client import MocogiClient

# Logging konfigurieren
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("migration")

def parse_markdown_table(file_path: str) -> List[Tuple[str, str]]:
    """
    Parst die Markdown-Datei und extrahiert die Modul-Mappings.
    Erwartet ein Format wie:
    | Modul in PO2 | Modul in PO3 |
    | Algorithmik | Algorithmen und Datenstrukturen |
    """
    mappings = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            # Ignoriere Header-Trenner (---) und Zeilen ohne Pipe
            if "|" not in line or "---" in line:
                continue

            # Teile Zeile an Pipes und säubere Felder
            parts = [p.strip() for p in line.split("|")]
            # Filtere leere Teile am Anfang/Ende durch split("|")
            parts = [p for p in parts if p]

            if len(parts) >= 2:
                po2_name = parts[0]
                po3_name = parts[1]

                # Ignoriere Header-Zeile
                if any(x in po2_name.upper() for x in ["PO2", "PO3", "MODUL"]):
                    continue

                mappings.append((po2_name, po3_name))

    except Exception as e:
        logger.error(f"Fehler beim Parsen der Datei {file_path}: {e}")

    return mappings

def map_to_protocol_update(full_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Konvertiert die vollständigen API-Daten in das ModuleProtocolUpdate Format.
    Erforderlich für die Mocogi API (PUT /moduleDrafts/{id}).
    """
    # Drafts haben oft eine andere Struktur als publizierte Module
    if "module" in full_data and "metadata" not in full_data:
        # Draft Struktur
        module_part = full_data.get("module", {})
        metadata = {
            "title": module_part.get("title"),
            "abbrev": module_part.get("abbreviation"),
            "moduleType": module_part.get("moduleType"),
            "ects": full_data.get("ects") or module_part.get("ects"),
            "language": module_part.get("language"),
            "duration": module_part.get("duration"),
            "season": module_part.get("season"),
            "workload": module_part.get("workload"),
            "status": module_part.get("status"),
            "location": module_part.get("location"),
            "participants": module_part.get("participants"),
            "moduleRelation": module_part.get("moduleRelation"),
            "management": module_part.get("management"),
            "lecturers": module_part.get("lecturers"),
            "assessmentMethods": module_part.get("assessmentMethods", {}),
            "examiner": module_part.get("examiner", {}),
            "examPhases": module_part.get("examPhases", []),
            "prerequisites": module_part.get("prerequisites", {}),
            "po": full_data.get("mandatoryPOs", []) + full_data.get("optionalPOs", []),
            "taughtWith": module_part.get("taughtWith", []),
            "attendanceRequirement": module_part.get("attendanceRequirement"),
            "assessmentPrerequisite": module_part.get("assessmentPrerequisite")
        }
        de_content = full_data.get("deContent") or module_part.get("deContent", {})
        en_content = full_data.get("enContent") or module_part.get("enContent", {})
    else:
        # Publizierte Struktur oder bereits standardisierte Struktur
        metadata = full_data.get("metadata", {}).copy()
        de_content = full_data.get("deContent", {})
        en_content = full_data.get("enContent", {})
        # Stelle sicher, dass abbrev -> abbreviation gemappt wird falls nötig
        if "abbreviation" in metadata and "abbrev" not in metadata:
            metadata["abbrev"] = metadata["abbreviation"]

    return {
        "metadata": metadata,
        "deContent": de_content,
        "enContent": en_content
    }

async def migrate_content(po2_id: str, po3_id: str, mappings: List[Tuple[str, str]]):
    """
    Führt die Migration der Inhalte durch.
    """
    async with MocogiClient() as client:
        # Authentifizierungs-Check
        logger.info("Prüfe Authentifizierung...")
        try:
            drafts = await client.get_module_drafts()
            logger.info(f"Authentifizierung erfolgreich. {len(drafts)} Modul-Entwürfe zugänglich.")
        except Exception as e:
            logger.warning(f"Authentifizierungs-Check fehlgeschlagen: {e}")
            logger.warning("Migration wird eventuell unvollständig sein (keine Drafts) oder bei Updates fehlschlagen.")

        # Verifiziere PO IDs und hole Studiengang-Namen
        logger.info("Verifiziere Prüfungsordnungen...")
        all_programs = await client.list_study_programs(filter="")

        po2_info = next((p for p in all_programs if p.get("po", {}).get("id") == po2_id), None)
        po3_info = next((p for p in all_programs if p.get("po", {}).get("id") == po3_id), None)

        if po2_info:
            logger.info(f"PO2 gefunden: {po2_info.get('deLabel')} (Version {po2_info.get('po', {}).get('version')})")
        else:
            logger.warning(f"PO2 ID '{po2_id}' nicht in API gefunden!")

        if po3_info:
            logger.info(f"PO3 gefunden: {po3_info.get('deLabel')} (Version {po3_info.get('po', {}).get('version')})")
        else:
            logger.warning(f"PO3 ID '{po3_id}' nicht in API gefunden!")

        logger.info(f"Lade Module für PO2: {po2_id}")
        po2_modules_list = await client.get_modules_by_po(po2_id)
        logger.info(f"  {len(po2_modules_list)} Module für PO2 geladen.")

        logger.info(f"Lade Module für PO3: {po3_id}")
        po3_modules_list = await client.get_modules_by_po(po3_id)
        logger.info(f"  {len(po3_modules_list)} Module für PO3 geladen.")

        # Mapping von Titel zu Modul-Objekt (vereinfacht die Suche)
        po2_by_title = {}
        for item in po2_modules_list:
            title = item.get("metadata", {}).get("title")
            if title:
                po2_by_title[title.lower().strip()] = item
                logger.debug(f"    Gefundenes PO2 Modul: {title} (Draft: {item.get('isDraft', False)})")

        po3_by_title = {}
        for item in po3_modules_list:
            title = item.get("metadata", {}).get("title")
            if title:
                po3_by_title[title.lower().strip()] = item
                logger.debug(f"    Gefundenes PO3 Modul: {title} (Draft: {item.get('isDraft', False)})")

        logger.info(f"Starte Migration von {len(mappings)} Mappings...")

        success_count = 0
        fail_count = 0

        for po2_name, po3_name in mappings:
            logger.info(f"Verarbeite: '{po2_name}' -> '{po3_name}'")

            # Suche Module in den geladenen Listen
            po2_key = po2_name.lower().strip()
            po3_key = po3_name.lower().strip()

            source_mod = po2_by_title.get(po2_key)
            target_mod = po3_by_title.get(po3_key)

            if not source_mod:
                logger.warning(f"  PO2 Modul '{po2_name}' nicht in API gefunden.")
                fail_count += 1
                continue

            if not target_mod:
                logger.warning(f"  PO3 Modul '{po3_name}' nicht in API gefunden.")
                fail_count += 1
                continue

            # Hole vollständige Details
            source_id = source_mod["id"]
            target_id = target_mod["id"]
            is_source_draft = source_mod.get("isDraft", False)
            is_target_draft = target_mod.get("isDraft", False)

            logger.info(f"  Hole Details für Source {source_id} (Draft: {is_source_draft}) und Target {target_id} (Draft: {is_target_draft})")

            if is_source_draft:
                full_source = await client.get_module_draft_details(source_id)
            else:
                full_source = await client.get_module_details(source_id)

            if is_target_draft:
                full_target = await client.get_module_draft_details(target_id)
            else:
                full_target = await client.get_module_details(target_id)

            # Transformation des payloads in das von der API erwartete Format
            payload = map_to_protocol_update(full_target)
            payload["deContent"] = full_source.get("deContent")
            payload["enContent"] = full_source.get("enContent")

            try:
                if is_target_draft:
                    await client.update_module_draft(target_id, payload)
                else:
                    await client.update_module(target_id, payload)
                logger.info(f"  Successfully migrated content for '{po3_name}'")
                success_count += 1
            except Exception as e:
                logger.error(f"  Fehler beim Update von '{po3_name}': {e}")
                fail_count += 1

        logger.info(f"Migration abgeschlossen. Erfolgreich: {success_count}, Fehlgeschlagen: {fail_count}")

async def main():
    parser = argparse.ArgumentParser(description="Migriert Modulinhalte zwischen POs.")
    parser.add_argument("md_file", help="Pfad zur Markdown-Datei mit der Äquivalenzliste")
    parser.add_argument("--po2", required=True, help="ID der alten PO (z.B. inf_mi4)")
    parser.add_argument("--po3", required=True, help="ID der neuen PO (z.B. inf_mi5)")

    args = parser.parse_args()

    mappings = parse_markdown_table(args.md_file)
    if not mappings:
        logger.error("Keine Mappings in der Datei gefunden.")
        return

    logger.info(f"{len(mappings)} Mappings gefunden.")
    await migrate_content(args.po2, args.po3, mappings)

if __name__ == "__main__":
    asyncio.run(main())
