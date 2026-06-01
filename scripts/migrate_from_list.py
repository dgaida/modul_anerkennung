"""Migriert Modulinhalte basierend auf einer Äquivalenzliste von PO2 nach PO3.

Dieses Skript liest eine Markdown-Tabelle (standardmäßig unter `data/aequivalenzliste.md`),
die Zuordnungen zwischen Modulen der Informatik PO2 (inf_inf2) und PO3 (inf_inf3) enthält.
Für jeden Eintrag wird der Inhalt (Deutsch/Englisch) vom PO2-Modul in den PO3-Entwurf (Draft)
übertragen, wobei die Metadaten des Ziels erhalten bleiben.

Nutzung:
    Das Skript wird typischerweise aus dem Wurzelverzeichnis des Projekts gestartet:
    `PYTHONPATH=. python3 scripts/migrate_from_list.py`

Ziel:
    Automatisierung der Befüllung von PO3-Modulentwürfen mit bewährten Inhalten aus der PO2.

Abhängigkeiten:
    - `scripts/standalone_restore_algo.py`: Enthält Kernfunktionen für API-Aufrufe und Mapping.
    - `data/aequivalenzliste.md`: Die Quelldatei für das Mapping.

Authentifizierung:
    Erfordert einen gültigen `MOCOGI_API_TOKEN` in der `secrets.env`.
"""

import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Importiere Funktionen aus dem Standalone-Skript
# Da das Skript keine .py Endung in sys.path braucht, wenn wir im richtigen Verzeichnis sind
# oder wir fügen den Pfad hinzu.
sys.path.append(str(Path(__file__).parent))
from standalone_restore_algo import api_call, get_module_by_title, get_draft_by_title, map_to_protocol_update

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("migrate_from_list")

def parse_markdown_table(file_path: str) -> List[Dict[str, Any]]:
    """Parst die Äquivalenzliste aus einer Markdown-Datei.

    Die Tabelle muss mindestens drei Spalten haben: Quelle (PO2), Ziel (PO3) und Semester.
    Header und Trennzeile werden übersprungen.

    Format-Beispiel:
    | Modul in PO2 | Modul in PO3 | Semester |
    | :--- | :--- | :--- |
    | Algorithmik | Algorithmen und Datenstrukturen | 2 |

    Args:
        file_path: Pfad zur Markdown-Datei mit der Äquivalenzliste.

    Returns:
        Eine Liste von Dictionaries mit den Schlüsseln 'source', 'target' und 'semester'.
    """
    entries = []
    if not os.path.exists(file_path):
        logger.error(f"Datei nicht gefunden: {file_path}")
        return entries

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Überspringe Header und Trennzeile
    for line in lines[2:]:
        line = line.strip()
        if not line or not line.startswith("|"):
            continue

        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 3:
            source_title = parts[0]
            target_title = parts[1]
            try:
                semester = int(parts[2])
            except ValueError:
                semester = 1

            entries.append({
                "source": source_title,
                "target": target_title,
                "semester": semester
            })

    return entries

def run_migration() -> None:
    """Führt die Migration basierend auf der Liste in data/aequivalenzliste.md aus.

    Der Prozess für jeden Eintrag umfasst:
    1. Abruf des Quell-Moduls aus der PO inf_inf2 via Titel.
    2. Abruf des Ziel-Entwurfs aus der PO inf_inf3 via Titel.
    3. Mergen der Daten (Inhalt von Quelle, Identität/Metadaten von Ziel).
    4. Senden des Updates (PUT) an den Mocogi-Draft-Endpunkt.

    Raises:
        Exception: Bei Fehlern während des API-Zugriffs oder der Datenverarbeitung.
    """
    list_path = "data/aequivalenzliste.md"
    base_url = "https://module.gm.th-koeln.de/api"

    logger.info(f"Starte Migration aus Liste: {list_path}")

    entries = parse_markdown_table(list_path)
    logger.info(f"{len(entries)} Einträge zum Verarbeiten gefunden.")

    success_count = 0
    fail_count = 0

    for entry in entries:
        source_title = entry["source"]
        target_title = entry["target"]
        semester = entry["semester"]

        logger.info(f"--- Verarbeite: '{source_title}' -> '{target_title}' (Sem: {semester}) ---")

        try:
            # 1. Quell-Modul finden (PO2)
            # Wir nehmen an, dass die Quelle in inf_inf2 liegt
            logger.info(f"Suche Quell-Modul '{source_title}' in PO inf_inf2...")
            source_data = get_module_by_title(base_url, "inf_inf2", source_title)
            logger.info(f"Quell-Modul '{source_title}' erfolgreich geladen (ID: {source_data.get('id')})")

            # 2. Ziel-Draft finden (PO3)
            logger.info(f"Suche Ziel-Draft '{target_title}' in PO inf_inf3...")
            target_data = get_draft_by_title(base_url, "inf_inf3", target_title)
            target_id = target_data.get('module', {}).get('id') or target_data.get('id')

            if not target_id:
                raise Exception(f"Konnte ID für Ziel-Draft '{target_title}' nicht ermitteln.")

            logger.info(f"Ziel-Draft '{target_title}' gefunden (Draft-ID: {target_data.get('id')}, Modul-ID: {target_id})")

            # 3. Payload erstellen
            logger.info(f"Erstelle Payload für '{target_title}'...")
            payload = map_to_protocol_update(source_data, target_data, recommended_semester=semester)

            # 4. Update durchführen
            target_url = f"{base_url}/moduleDrafts/{target_id}"
            headers = {"Mocogi-Version-Scheme": "v1.0s"}

            logger.info(f"Sende Update für '{target_title}' an {target_url}...")
            api_call(target_url, method="PUT", data=payload, extra_headers=headers)

            logger.info(f"Erfolgreich migriert: {target_title}")
            success_count += 1

        except Exception as e:
            logger.error(f"Fehler bei Migration von '{target_title}': {e}")
            fail_count += 1

    logger.info("========================================")
    logger.info(f"Migration abgeschlossen. Erfolg: {success_count}, Fehler: {fail_count}")

if __name__ == "__main__":
    run_migration()
