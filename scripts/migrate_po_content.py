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
    """
    mappings = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if "|" not in line or "---" in line:
                continue

            parts = [p.strip() for p in line.split("|")]
            parts = [p for p in parts if p]

            if len(parts) >= 2:
                po2_name = parts[0]
                po3_name = parts[1]
                if any(x in po2_name.upper() for x in ["PO2", "PO3", "MODUL"]):
                    continue
                mappings.append((po2_name, po3_name))
    except Exception as e:
        logger.error(f"Fehler beim Parsen der Datei {file_path}: {e}")
    return mappings

def map_to_protocol_update(full_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Konvertiert die vollständigen API-Daten in das ModuleProtocolUpdate Format.
    """
    # Drafts haben oft eine andere Struktur als publizierte Module
    if "module" in full_data and not "metadata" in full_data:
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
        logger.info("Lade Module für PO2 und PO3...")
        po2_modules = await client.get_modules_by_po(po2_id)
        po3_modules = await client.get_modules_by_po(po3_id)

        po2_by_title = {m["metadata"]["title"].lower().strip(): m for m in po2_modules}
        po3_by_title = {m["metadata"]["title"].lower().strip(): m for m in po3_modules}

        success_count = 0
        fail_count = 0

        for po2_name, po3_name in mappings:
            logger.info(f"Verarbeite: '{po2_name}' -> '{po3_name}'")
            source_mod = po2_by_title.get(po2_name.lower().strip())
            target_mod = po3_by_title.get(po3_name.lower().strip())

            if not source_mod or not target_mod:
                logger.warning(f"  Modul nicht gefunden. Source: {bool(source_mod)}, Target: {bool(target_mod)}")
                fail_count += 1
                continue

            # Details holen
            if source_mod["isDraft"]:
                full_source = await client.get_module_draft_details(source_mod["id"])
            else:
                full_source = await client.get_module_details(source_mod["id"])

            if target_mod["isDraft"]:
                full_target = await client.get_module_draft_details(target_mod["id"])
            else:
                full_target = await client.get_module_details(target_mod["id"])

            # Transformation
            payload = map_to_protocol_update(full_target)
            payload["deContent"] = full_source.get("deContent")
            payload["enContent"] = full_source.get("enContent")

            try:
                if target_mod["isDraft"]:
                    await client.update_module_draft(target_mod["id"], payload)
                else:
                    await client.update_module(target_mod["id"], payload)
                logger.info(f"  Erfolgreich migriert: '{po3_name}'")
                success_count += 1
            except Exception as e:
                logger.error(f"  Fehler beim Update von '{po3_name}': {e}")
                fail_count += 1

        logger.info(f"Abgeschlossen. Erfolgreich: {success_count}, Fehlgeschlagen: {fail_count}")

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("md_file")
    parser.add_argument("--po2", required=True)
    parser.add_argument("--po3", required=True)
    args = parser.parse_args()
    mappings = parse_markdown_table(args.md_file)
    await migrate_content(args.po2, args.po3, mappings)

if __name__ == "__main__":
    asyncio.run(main())
