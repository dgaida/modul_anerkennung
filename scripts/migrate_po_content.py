#!/usr/bin/env python3
"""
Skript zur Migration von Modulinhalten zwischen zwei Prüfungsordnungen (PO).
Basierend auf einer Äquivalenzliste in einer Markdown-Datei.
"""

import asyncio
import argparse
import logging
import re
from typing import Dict, List, Tuple
from modul_anerkennung.mcp_client import MocogiClient

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
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
                if "PO2" in po2_name.upper() or "PO3" in po3_name.upper() or "MODUL" in po2_name.upper():
                    continue

                mappings.append((po2_name, po3_name))

    except Exception as e:
        logger.error(f"Fehler beim Parsen der Datei {file_path}: {e}")

    return mappings

async def migrate_content(po2_id: str, po3_id: str, mappings: List[Tuple[str, str]]):
    """
    Führt die Migration der Inhalte durch.
    """
    async with MocogiClient() as client:
        logger.info(f"Lade Module für PO2: {po2_id}")
        po2_modules_list = await client.get_modules_by_po(po2_id)

        logger.info(f"Lade Module für PO3: {po3_id}")
        po3_modules_list = await client.get_modules_by_po(po3_id)

        # Mapping von Titel zu Modul-Objekt (vereinfacht die Suche)
        # Note: Mocogi returns a list of dicts where each dict has a 'module' key
        po2_by_title = {}
        for item in po2_modules_list:
            mod = item.get('module', {})
            title = mod.get('metadata', {}).get('title')
            if title:
                po2_by_title[title] = mod

        po3_by_title = {}
        for item in po3_modules_list:
            mod = item.get('module', {})
            title = mod.get('metadata', {}).get('title')
            if title:
                po3_by_title[title] = mod

        logger.info(f"Starte Migration von {len(mappings)} Mappings...")

        success_count = 0
        fail_count = 0

        for po2_name, po3_name in mappings:
            logger.info(f"Verarbeite: '{po2_name}' -> '{po3_name}'")

            # Suche Module in den geladenen Listen
            source_mod = po2_by_title.get(po2_name)
            target_mod = po3_by_title.get(po3_name)

            if not source_mod:
                logger.warning(f"  PO2 Modul '{po2_name}' nicht in API gefunden.")
                fail_count += 1
                continue

            if not target_mod:
                logger.warning(f"  PO3 Modul '{po3_name}' nicht in API gefunden.")
                fail_count += 1
                continue

            # Hole vollständige Details (da die Liste nur Metadaten enthalten könnte)
            source_id = source_mod['id']
            target_id = target_mod['id']

            logger.info(f"  Hole Details für Source {source_id} und Target {target_id}")
            full_source = await client.get_module_details(source_id)
            full_target = await client.get_module_details(target_id)

            # Kopiere Inhalte
            # Wir behalten die Metadaten von PO3 bei, kopieren aber deContent und enContent
            updated_data = full_target.copy()
            updated_data['deContent'] = full_source.get('deContent')
            updated_data['enContent'] = full_source.get('enContent')

            try:
                await client.update_module(target_id, updated_data)
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
