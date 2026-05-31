import json
import logging
import os
import pathlib
import sys
import urllib.request
import urllib.error

# Logging-Konfiguration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("standalone_restore")


def load_env_manual():
    """Lädt Umgebungsvariablen aus .env oder secrets.env manuell."""
    possible_paths = [
        pathlib.Path.cwd() / "secrets.env",
        pathlib.Path.cwd() / ".env",
        pathlib.Path(__file__).resolve().parent.parent / "secrets.env",
        pathlib.Path(__file__).resolve().parent.parent / ".env",
    ]

    found = False
    for path in possible_paths:
        if path.exists():
            logger.info(f"Lade Umgebungsvariablen aus {path}")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)

                        # Säubere Key (entferne 'export ' und Leerzeichen)
                        key = key.strip()
                        if key.lower().startswith("export "):
                            key = key[7:].strip()

                        # Säubere Value (entferne Inline-Kommentare, Leerzeichen und Anführungszeichen)
                        if "#" in value:
                            # Vorsicht: # innerhalb von Anführungszeichen sollte eigentlich erhalten bleiben,
                            # aber .env Dateien sind meist simpel. Wir nehmen an, # startet einen Kommentar.
                            value = value.split("#", 1)[0]

                        value = value.strip().strip("'").strip('"')

                        os.environ[key] = value
            found = True
            break

    if not found:
        logger.warning("Keine .env oder secrets.env Datei gefunden.")

    # Alias Handling
    if not os.getenv("MOCOGI_API_TOKEN") and os.getenv("MOCOGI_API_KEY"):
        os.environ["MOCOGI_API_TOKEN"] = os.environ["MOCOGI_API_KEY"]

    token = os.getenv("MOCOGI_API_TOKEN")
    if token:
        masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
        logger.info(f"MOCOGI_API_TOKEN gefunden: {masked}")
    else:
        logger.warning("MOCOGI_API_TOKEN wurde nicht gefunden!")

load_env_manual()

def api_call(url, method="GET", data=None, extra_headers=None):
    """Führt einen API-Call mit urllib aus."""
    logger.debug(f"API Call: {method} {url}")

    headers = {
        "User-Agent": "Standalone-Restore-Script/1.0"
    }

    token = os.getenv("MOCOGI_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
        masked_token = token[:4] + "..." + token[-4:] if len(token) > 8 else "***"
        logger.debug(f"Sende Authorization Header mit Token {masked_token}")
    else:
        logger.warning("Kein MOCOGI_API_TOKEN vorhanden. Anfrage erfolgt ohne Authentifizierung.")

    if extra_headers:
        headers.update(extra_headers)
        logger.debug(f"Zusätzliche Header: {list(extra_headers.keys())}")

    payload = None
    if data:
        payload = json.dumps(data).encode("utf-8")
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=payload, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            body = response.read().decode("utf-8")
            logger.debug(f"Antwort Status: {status}")
            if body:
                return json.loads(body)
            return {}
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        logger.error(f"HTTP Fehler: {e.code} - {e.reason}")
        logger.error(f"Fehler-Body: {error_body}")

        # Log headers sent (except full token)
        sent_headers = headers.copy()
        if "Authorization" in sent_headers:
            sent_headers["Authorization"] = "Bearer " + (token[:4] + "..." if token else "***")
        logger.debug(f"Gesendete Header: {sent_headers}")

        msg = f"API Call fehlgeschlagen: {e.code} {e.reason}"
        if e.code in [401, 403]:
            msg += " (Hinweis: MOCOGI_API_TOKEN könnte abgelaufen sein!)"
        raise Exception(msg)
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim API-Call: {e}")
        raise


def get_draft_by_title(base_url, po_id, title):
    """Sucht einen Modul-Draft anhand von PO-ID und Titel."""
    url = f"{base_url}/moduleDrafts"
    logger.info(f"Lade alle Entwürfe von {url}...")
    data = api_call(url)

    # Extrahiere direct und indirect Drafts (siehe Memory)
    drafts = (data.get("direct") or []) + (data.get("indirect") or [])
    logger.debug(f"{len(drafts)} Entwürfe insgesamt gefunden.")

    search_title = title.lower().strip()
    for draft in drafts:
        mandatory = draft.get("mandatoryPOs") or []
        optional = draft.get("optionalPOs") or []

        if po_id in mandatory or po_id in optional:
            module_info = draft.get("module") or {}
            # Manchmal ist der Titel auch direkt im Draft-Objekt oder tiefer verschachtelt
            draft_title = (module_info.get("title") or
                           draft.get("title") or
                           (module_info.get("metadata") or {}).get("title") or
                           "")

            if draft_title.lower().strip() == search_title:
                logger.info(f"Passenden Entwurf gefunden: {draft_title} (ID: {draft.get('id')})")
                return draft

    raise Exception(f"Kein Entwurf mit dem Titel {title} in PO {po_id} gefunden. (Hinweis: MOCOGI_API_TOKEN könnte abgelaufen sein!)")

def map_to_protocol_update(source_data: dict, target_data: dict) -> dict:
    """Konvertiert die API-Daten in das ModuleProtocolUpdate Format.
    Dabei werden die Metadaten vom Ziel-Draft (target_data) verwendet und die
    Inhalte (deContent, enContent) vom Quell-Modul (source_data).
    """
    logger.debug("Mappe Quelldaten auf Zielformat unter Verwendung von Ziel-Metadaten...")

    # Extrahiere Metadaten aus dem Ziel-Draft
    # Laut Memory: Metadaten unter "module", ects und POs auf Top-Level
    tm = target_data.get("module") or {}
    tm_meta = tm.get("metadata") or tm

    # Fallback auf Source-Metadaten falls Target unvollständig
    sm = (source_data.get("metadata") or
          source_data.get("module", {}).get("metadata") or
          source_data.get("module") or {})

    # Mapping auf die Struktur von ModuleProtocolUpdate
    metadata = {
        "title": tm_meta.get("title") or "Algorithmen und Datenstrukturen",
        "abbrev": tm_meta.get("abbreviation") or tm_meta.get("abbrev") or sm.get("abbreviation") or "Algo",
        "moduleType": tm_meta.get("moduleType") or sm.get("moduleType", "module"),
        "ects": target_data.get("ects") or tm_meta.get("ects") or source_data.get("ects") or 6,
        "language": tm_meta.get("language") or sm.get("language", "de"),
        "duration": tm_meta.get("duration") or sm.get("duration", 1),
        "season": tm_meta.get("season") or sm.get("season", "ss"),
        "workload": tm_meta.get("workload") or sm.get("workload") or {
            "lecture": 0, "seminar": 0, "practical": 0,
            "exercise": 0, "projectSupervision": 0, "projectWork": 0
        },
        "status": "active",
        "location": tm_meta.get("location") or sm.get("location", "gm"),
        "participants": tm_meta.get("participants") or sm.get("participants"),
        "moduleRelation": tm_meta.get("moduleRelation") or sm.get("moduleRelation"),
        "moduleManagement": tm_meta.get("moduleManagement") or tm_meta.get("management") or sm.get("moduleManagement") or [],
        "lecturers": tm_meta.get("lecturers") or sm.get("lecturers") or [],
        "assessmentMethods": {
            "mandatory": (tm_meta.get("assessmentMethods") or {}).get("mandatory") or (sm.get("assessmentMethods") or {}).get("mandatory") or []
        },
        "examiner": {
            "first": (tm_meta.get("examiner") or {}).get("first") or (sm.get("examiner") or {}).get("first"),
            "second": (tm_meta.get("examiner") or {}).get("second") or (sm.get("examiner") or {}).get("second")
        },
        "examPhases": tm_meta.get("examPhases") or sm.get("examPhases") or [],
        "prerequisites": {
            "recommended": (tm_meta.get("prerequisites") or {}).get("recommended") or (sm.get("prerequisites") or {}).get("recommended") or {"text": "", "modules": []},
            "required": (tm_meta.get("prerequisites") or {}).get("required") or (sm.get("prerequisites") or {}).get("required")
        },
        "po": target_data.get("mandatoryPOs") or target_data.get("optionalPOs") or ["inf_inf3"],
        "taughtWith": tm_meta.get("taughtWith") or sm.get("taughtWith") or [],
        "attendanceRequirement": tm_meta.get("attendanceRequirement") or sm.get("attendanceRequirement"),
        "assessmentPrerequisite": tm_meta.get("assessmentPrerequisite") or sm.get("assessmentPrerequisite")
    }

    payload = {
        "metadata": metadata,
        "deContent": source_data.get("deContent") or sm.get("deContent") or {},
        "enContent": source_data.get("enContent") or sm.get("enContent") or {}
    }

    logger.debug(f"Payload erstellt: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    return payload


def restore():
    """Hauptfunktion zur Wiederherstellung des Algo-Moduls."""
    # ID eines funktionsfähigen Algo-Moduls (hier aus inf_inf2)
    source_id = "21723454-3c3e-4ebe-ade0-82eacb69b185"

    # Ziel-Modul in inf_inf3
    target_po = "inf_inf3"
    target_title = "Algorithmen und Datenstrukturen"

    base_url = "https://module.gm.th-koeln.de/api"

    logger.info(f"Starte Wiederherstellung von '{target_title}' in {target_po}...")

    try:
        # 1. Lade Quelldaten
        source_url = f"{base_url}/modules/{source_id}"
        logger.info(f"Lade Quelldaten von {source_url}...")
        source_data = api_call(source_url)

        # 2. Lade Zieldaten (Draft) um Metadaten zu erhalten
        # Wir suchen den Draft via Titel, da die ID allein oft nicht ausreicht oder sich ändern kann
        target_data = get_draft_by_title(base_url, target_po, target_title)
        target_id = target_data.get('id')

        # 3. Mappe Daten (Merge)
        payload = map_to_protocol_update(source_data, target_data)

        # 4. Update Draft
        target_url = f"{base_url}/moduleDrafts/{target_id}"
        logger.info(f"Sende Update an {target_url}...")
        headers = {"Mocogi-Version-Scheme": "v1.0s"}
        result = api_call(target_url, method="PUT", data=payload, extra_headers=headers)

        logger.info("Wiederherstellung erfolgreich abgeschlossen!")
        logger.debug(f"Ergebnis: {json.dumps(result, indent=2, ensure_ascii=False)}")

    except Exception as e:
        logger.error(f"Fehler bei der Wiederherstellung: {e}")
        logger.error("Hinweis: Stellen Sie sicher, dass der MOCOGI_API_TOKEN aktuell ist und Schreibrechte besitzt.")
        sys.exit(1)

if __name__ == "__main__":
    restore()
