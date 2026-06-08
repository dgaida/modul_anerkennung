"""
Erzeugt ein Word-Dokument mit einer Äquivalenzliste zwischen PO2 und PO3.

Dieses Skript liest die Äquivalenzliste aus `data/aequivalenzliste.md` und ergänzt sie
um alle weiteren Module aus der Informatik PO2 (inf_inf2) und PO3 (inf_inf3).
Das Ergebnis ist eine 4-spaltige Tabelle (PO2 Name, PO2 ECTS, PO3 Name, PO3 ECTS),
die nach Semester und Titel sortiert ist. Äquivalente Module stehen in derselben Zeile.

Nutzung:
    PYTHONPATH=. python3 scripts/create_equivalence_table_word.py
"""

import os
import json
import urllib.request
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("equivalence_table")

def load_env() -> None:
    """Lädt Umgebungsvariablen aus secrets.env oder .env.

    Berücksichtigt das 'export ' Präfix und entfernt Kommentare.
    """
    for env_file in [Path("secrets.env"), Path(".env")]:
        if env_file.exists():
            logger.info(f"Lade Umgebungsvariablen aus {env_file}")
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("export "):
                        line = line[7:]
                    # Entferne Inline-Kommentare
                    line = line.split("#")[0].strip()
                    key_val = line.split("=", 1)
                    if len(key_val) == 2:
                        key = key_val[0].strip()
                        val = key_val[1].strip().strip("'").strip('"')
                        os.environ[key] = val

load_env()

API_TOKEN = os.getenv("MOCOGI_API_TOKEN") or os.getenv("MOCOGI_API_KEY")
BASE_URL = "https://module.gm.th-koeln.de/api"

def api_call(url: str, method: str = "GET", data: Any = None) -> Any:
    """Führt einen API-Aufruf an die Mocogi API durch.

    Args:
        url: Die Ziel-URL.
        method: HTTP-Methode (GET, PUT, etc.).
        data: Optionaler JSON-Body.

    Returns:
        Die geparsten JSON-Daten oder None bei Authentifizierungsfehlern.

    Raises:
        urllib.error.HTTPError: Bei anderen API-Fehlern.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    req = urllib.request.Request(url, method=method, headers=headers)
    if data:
        json_data = json.dumps(data).encode("utf-8")
        req.data = json_data

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        # Logge Status und Body für Debugging (wie in anderen Skripten erwünscht)
        logger.debug(f"API Fehler {e.code} für {url}: {body}")
        if e.code in [401, 403]:
            logger.warning(f"Authentifizierungsfehler (401/403) bei {url}. Token prüfen.")
            return None
        raise
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim API-Aufruf ({url}): {e}")
        raise

def get_module_metadata(m: Dict[str, Any]) -> Dict[str, Any]:
    """Extrahiert Metadaten robust aus verschiedenen Verschachtelungsebenen.

    Args:
        m: Das Modul- oder Draft-Objekt von der API.

    Returns:
        Das Dictionary mit den Metadaten.
    """
    # Priorität: top-level metadata > module.metadata > module (als metadata)
    metadata = m.get("metadata") or m.get("module", {}).get("metadata") or m.get("module") or {}
    if not isinstance(metadata, dict):
        return {}
    return metadata

def get_module_title(m: Dict[str, Any]) -> str:
    """Extrahiert den Titel aus einem Modul- oder Draft-Objekt.

    Args:
        m: Das Modul- oder Draft-Objekt.

    Returns:
        Der Titel des Moduls.
    """
    meta = get_module_metadata(m)
    title = meta.get("title") or m.get("title") or ""
    return str(title).strip()

def get_module_ects(m: Dict[str, Any]) -> str:
    """Extrahiert die ECTS aus einem Modul- oder Draft-Objekt.

    Args:
        m: Das Modul- oder Draft-Objekt.

    Returns:
        Die ECTS-Punkte als String.
    """
    meta = get_module_metadata(m)
    # ECTS stehen bei Drafts oft auf Top-Level, bei Modulen in Metadata
    ects = m.get("ects") or meta.get("ects") or ""
    return str(ects)

def get_semester(m: Dict[str, Any], po_id: str) -> int:
    """Extrahiert das empfohlene Semester für eine bestimmte PO.

    Args:
        m: Das Modul- oder Draft-Objekt.
        po_id: Die Kennung der Prüfungsordnung (z.B. 'inf_inf2').

    Returns:
        Das Semester als Integer oder 99 als Fallback.
    """
    meta = get_module_metadata(m)
    po_info = meta.get("po") or {}

    # Prüfe mandatory und optional Listen
    for category in ["mandatory", "optional"]:
        items = po_info.get(category) or []
        for item in items:
            if item.get("po") == po_id:
                sem_list = item.get("recommendedSemester") or []
                if sem_list:
                    return int(sem_list[0])

    return 99

def parse_markdown_table(file_path: str) -> List[Dict[str, Any]]:
    """Parst die Äquivalenzliste aus einer Markdown-Datei.

    Args:
        file_path: Pfad zur MD-Datei.

    Returns:
        Liste von Dictionaries mit 'source', 'target' und 'semester'.
    """
    entries = []
    if not os.path.exists(file_path):
        logger.error(f"Datei nicht gefunden: {file_path}")
        return entries

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or not line.startswith("|") or "Modul in PO" in line or "| :---" in line:
            continue

        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 2:
            source_title = parts[0]
            target_title = parts[1]
            semester = 0
            if len(parts) >= 3:
                try:
                    semester = int(parts[2])
                except ValueError:
                    semester = 0

            entries.append({
                "source": source_title,
                "target": target_title,
                "semester": semester
            })
    return entries

def main() -> None:
    """Hauptfunktion zur Tabellenerzeugung."""
    logger.info("Starte Generierung der Äquivalenzliste für Word...")

    # 1. Daten von API abrufen
    logger.info("Lade Module der PO2 (inf_inf2)...")
    # API erfordert select=metadata für PO-Filterung wie in anderen Skripten
    po2_raw = api_call(f"{BASE_URL}/modules?po=inf_inf2&active=true&select=metadata") or []
    logger.info(f"{len(po2_raw)} Module in PO2 gefunden.")

    logger.info("Lade Entwürfe der PO3 (inf_inf3)...")
    drafts_resp = api_call(f"{BASE_URL}/moduleDrafts")
    po3_raw = []
    if drafts_resp:
        all_drafts = (drafts_resp.get("direct") or []) + (drafts_resp.get("indirect") or [])
        # Filter auf inf_inf3 (Mandatory oder Optional)
        po3_raw = [d for d in all_drafts if "inf_inf3" in (d.get("mandatoryPOs") or []) or "inf_inf3" in (d.get("optionalPOs") or [])]
    else:
        logger.warning("Konnte Entwürfe nicht laden (Token abgelaufen?).")
    logger.info(f"{len(po3_raw)} Module/Entwürfe in PO3 gefunden.")

    # 2. Äquivalenzliste einlesen
    list_path = "data/aequivalenzliste.md"
    equiv_list = parse_markdown_table(list_path)
    logger.info(f"{len(equiv_list)} Einträge aus {list_path} geladen.")

    # 3. Hilfs-Mappings erstellen (Titel -> Modul-Objekt)
    po2_by_title = {get_module_title(m).lower(): m for m in po2_raw}
    po3_by_title = {get_module_title(m).lower(): m for m in po3_raw}

    used_po2: Set[str] = set()
    used_po3: Set[str] = set()
    rows: List[Dict[str, Any]] = []

    # 4. Einträge aus der Äquivalenzliste verarbeiten (Zusammenführung)
    for eq in equiv_list:
        src_t_raw = eq["source"]
        tgt_t_raw = eq["target"]
        src_t = src_t_raw.lower()
        tgt_t = tgt_t_raw.lower()

        m2 = po2_by_title.get(src_t)
        m3 = po3_by_title.get(tgt_t)

        if m2:
            used_po2.add(src_t)
        if m3:
            used_po3.add(tgt_t)

        # Bestimme Sortier-Semester
        sort_sem = eq["semester"]
        if not sort_sem:
            # Prio: PO3 Semester, dann PO2 Semester
            sort_sem = get_semester(m3, "inf_inf3") if m3 else get_semester(m2, "inf_inf2")

        rows.append({
            "po2": m2,
            "po2_title_fallback": src_t_raw if not m2 else None,
            "po3": m3,
            "po3_title_fallback": tgt_t_raw if not m3 else None,
            "sort_semester": sort_sem,
            "sort_title": (get_module_title(m3) if m3 else (get_module_title(m2) if m2 else tgt_t_raw))
        })

    # 5. Restliche PO2 Module hinzufügen (ohne PO3-Äquivalent in der Liste)
    for title_lower, m in po2_by_title.items():
        if title_lower not in used_po2:
            rows.append({
                "po2": m,
                "po3": None,
                "sort_semester": get_semester(m, "inf_inf2"),
                "sort_title": get_module_title(m)
            })

    # 6. Restliche PO3 Module hinzufügen (ohne PO2-Äquivalent in der Liste)
    for title_lower, m in po3_by_title.items():
        if title_lower not in used_po3:
            rows.append({
                "po2": None,
                "po3": m,
                "sort_semester": get_semester(m, "inf_inf3"),
                "sort_title": get_module_title(m)
            })

    # 7. Sortierung: nach Semester, dann alphabetisch nach Titel
    rows.sort(key=lambda x: (x["sort_semester"], x["sort_title"].lower()))

    # 8. Word-Dokument erstellen
    doc = Document()

    # Seitenränder für Tabellen optimieren
    for section in doc.sections:
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)

    doc.add_heading('Anlage: Äquivalenzliste PO2 / PO3 Informatik', 0)

    # Tabelle mit 4 Spalten erzeugen
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'

    # Header setzen
    hdr_cells = table.rows[0].cells
    header_labels = ['Modul PO2', 'ECTS PO2', 'Modul PO3', 'ECTS PO3']
    for i, label in enumerate(header_labels):
        p = hdr_cells[i].paragraphs[0]
        run = p.add_run(label)
        run.bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Datenzeilen befüllen
    for row_data in rows:
        row_cells = table.add_row().cells

        # PO2 Spalten (1 & 2)
        m2 = row_data["po2"]
        if m2:
            row_cells[0].text = get_module_title(m2)
            row_cells[1].text = get_module_ects(m2)
        elif row_data.get("po2_title_fallback"):
            row_cells[0].text = row_data["po2_title_fallback"]
            row_cells[1].text = "-"
        else:
            # Leerzeile zur Ausrichtung (User-Wunsch)
            row_cells[0].text = ""
            row_cells[1].text = ""

        # PO3 Spalten (3 & 4)
        m3 = row_data["po3"]
        if m3:
            row_cells[2].text = get_module_title(m3)
            row_cells[3].text = get_module_ects(m3)
        elif row_data.get("po3_title_fallback"):
            row_cells[2].text = row_data["po3_title_fallback"]
            row_cells[3].text = "-"
        else:
            # Leerzeile zur Ausrichtung
            row_cells[2].text = ""
            row_cells[3].text = ""

        # ECTS-Spalten zentrieren
        row_cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        row_cells[3].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    output_file = "aequivalenzliste_po2_po3.docx"
    doc.save(output_file)
    logger.info(f"Dokument erfolgreich unter '{output_file}' gespeichert.")
    print(f"\nERFOLG: {output_file} mit {len(rows)} Zeilen erstellt.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fehler bei der Ausführung: {e}")
        sys.exit(1)
