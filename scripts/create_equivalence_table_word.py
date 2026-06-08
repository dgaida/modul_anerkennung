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
import logging
import sys
import urllib.request
import urllib.error
from typing import List, Dict, Any, Set
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Konfiguration und Environment laden
import modul_anerkennung.config  # noqa: F401

BASE_URL = "https://module.gm.th-koeln.de/api"
API_TOKEN = os.environ.get("MOCOGI_API_TOKEN") or os.environ.get("MOCOGI_API_KEY")

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("create_word_table")

def api_call(url: str, method: str = "GET", data: Dict[str, Any] = None) -> Any:
    """Führt einen API-Aufruf aus und handhabt Authentifizierungsfehler.

    Args:
        url: Die URL des API-Endpunkts.
        method: Die HTTP-Methode (Standard: "GET").
        data: Optionales Dictionary mit JSON-Daten für den Body.

    Returns:
        Die Antwort der API als Dictionary oder None bei Auth-Fehlern.

    Raises:
        urllib.error.HTTPError: Bei anderen API-Fehlern (z.B. 404, 500).
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

def is_mandatory(m: Dict[str, Any], po_id: str) -> bool:
    """Prüft, ob ein Modul für eine bestimmte PO ein Pflichtmodul ist.

    Args:
        m: Das Modul- oder Draft-Objekt.
        po_id: Die Kennung der Prüfungsordnung (z.B. 'inf_inf2').

    Returns:
        True, wenn es ein Pflichtmodul ist, sonst False.
    """
    # Fall 1: Draft-Objekt (PO3)
    mandatory_pos = m.get("mandatoryPOs") or []
    if po_id in mandatory_pos:
        return True

    # Fall 2: Publiziertes Modul (PO2) - Metadaten Struktur
    meta = get_module_metadata(m)
    po_info = meta.get("po") or {}
    mandatory_list = po_info.get("mandatory") or []
    for item in mandatory_list:
        if item.get("po") == po_id:
            return True

    return False

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
    logger.info("Starte Generierung der Äquivalenzliste für Word (nur Pflichtmodule)...")

    # 1. Daten von API abrufen (alle aktiven für Lookup)
    logger.info("Lade Module der PO2 (inf_inf2)...")
    po2_all = api_call(f"{BASE_URL}/modules?po=inf_inf2&active=true&select=metadata") or []

    logger.info("Lade Entwürfe der PO3 (inf_inf3)...")
    drafts_resp = api_call(f"{BASE_URL}/moduleDrafts")
    po3_all = []
    if drafts_resp:
        po3_all = (drafts_resp.get("direct") or []) + (drafts_resp.get("indirect") or [])
    else:
        logger.warning("Konnte Entwürfe nicht laden (Token abgelaufen?).")

    # 2. Äquivalenzliste einlesen
    list_path = "data/aequivalenzliste.md"
    equiv_list = parse_markdown_table(list_path)
    logger.info(f"{len(equiv_list)} Einträge aus {list_path} geladen.")

    # 3. Hilfs-Mappings erstellen (Titel -> Modul-Objekt)
    po2_lookup = {get_module_title(m).lower(): m for m in po2_all}
    po3_lookup = {get_module_title(m).lower(): m for m in po3_all}

    used_po2: Set[str] = set()
    used_po3: Set[str] = set()
    rows: List[Dict[str, Any]] = []

    # 4. Einträge aus der Äquivalenzliste verarbeiten (Zusammenführung)
    for eq in equiv_list:
        src_t_raw = eq["source"]
        tgt_t_raw = eq["target"]
        src_t = src_t_raw.lower()
        tgt_t = tgt_t_raw.lower()

        m2 = po2_lookup.get(src_t)
        m3 = po3_lookup.get(tgt_t)

        # Filter: Mindestens eines muss mandatory sein
        is_m2_man = m2 and is_mandatory(m2, "inf_inf2")
        is_m3_man = m3 and is_mandatory(m3, "inf_inf3")

        if not is_m2_man and not is_m3_man:
            logger.debug(f"Überspringe Wahl-Äquivalenz: {src_t_raw} -> {tgt_t_raw}")
            continue

        if is_m2_man:
            used_po2.add(src_t)
        if is_m3_man:
            used_po3.add(tgt_t)

        # Bestimme Sortier-Semester
        sort_sem = eq["semester"] or (get_semester(m3, "inf_inf3") if m3 else get_semester(m2, "inf_inf2"))

        rows.append({
            "po2": m2,
            "po2_title_fallback": src_t_raw if not m2 else None,
            "po3": m3,
            "po3_title_fallback": tgt_t_raw if not m3 else None,
            "sort_semester": sort_sem,
            "sort_title": (get_module_title(m3) if m3 else (get_module_title(m2) if m2 else tgt_t_raw))
        })

    # 5. Restliche Pflichtmodule hinzufügen (ohne PO3-Äquivalent in der Liste)
    for title_lower, m in po2_lookup.items():
        if is_mandatory(m, "inf_inf2") and title_lower not in used_po2:
            rows.append({
                "po2": m,
                "po3": None,
                "sort_semester": get_semester(m, "inf_inf2"),
                "sort_title": get_module_title(m)
            })

    # 6. Restliche Pflichtmodule hinzufügen (ohne PO2-Äquivalent in der Liste)
    for title_lower, m in po3_lookup.items():
        if is_mandatory(m, "inf_inf3") and title_lower not in used_po3:
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
