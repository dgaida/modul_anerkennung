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
                        # Entferne Anführungszeichen falls vorhanden
                        value = value.strip().strip("'").strip('"')
                        os.environ[key.strip()] = value
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

    if extra_headers:
        headers.update(extra_headers)

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
        raise Exception(f"API Call fehlgeschlagen: {e.code} {e.reason}")
    except Exception as e:
        logger.error(f"Unerwarteter Fehler beim API-Call: {e}")
        raise


def map_to_protocol_update(source_data: dict) -> dict:
    """Konvertiert die API-Daten in das ModuleProtocolUpdate Format."""
    logger.debug("Mappe Quelldaten auf Zielformat...")

    # Extrahiere Metadaten aus verschiedenen möglichen Ebenen
    m = (source_data.get("metadata") or
         source_data.get("module", {}).get("metadata") or
         source_data.get("module") or {})

    # Mapping auf die Struktur von ModuleProtocolUpdate
    metadata = {
        "title": m.get("title") or "Algorithmen und Datenstrukturen",
        "abbrev": m.get("abbreviation") or m.get("abbrev") or "Algo",
        "moduleType": m.get("moduleType", "module"),
        "ects": source_data.get("ects") or m.get("ects") or 6,
        "language": m.get("language", "de"),
        "duration": m.get("duration", 1),
        "season": m.get("season", "ss"),
        "workload": m.get("workload") or {
            "lecture": 0, "seminar": 0, "practical": 0,
            "exercise": 0, "projectSupervision": 0, "projectWork": 0
        },
        "status": "active",
        "location": m.get("location", "gm"),
        "participants": m.get("participants"),
        "moduleRelation": m.get("moduleRelation"),
        "moduleManagement": m.get("moduleManagement") or m.get("management") or [],
        "lecturers": m.get("lecturers") or [],
        "assessmentMethods": {
            "mandatory": (m.get("assessmentMethods") or {}).get("mandatory") or []
        },
        "examiner": {
            "first": (m.get("examiner") or {}).get("first"),
            "second": (m.get("examiner") or {}).get("second")
        },
        "examPhases": m.get("examPhases") or [],
        "prerequisites": {
            "recommended": (m.get("prerequisites") or {}).get("recommended") or {"text": "", "modules": []},
            "required": (m.get("prerequisites") or {}).get("required")
        },
        "po": ["inf_inf3"],
        "taughtWith": m.get("taughtWith") or [],
        "attendanceRequirement": m.get("attendanceRequirement"),
        "assessmentPrerequisite": m.get("assessmentPrerequisite")
    }

    payload = {
        "metadata": metadata,
        "deContent": source_data.get("deContent") or m.get("deContent") or {},
        "enContent": source_data.get("enContent") or m.get("enContent") or {}
    }

    logger.debug(f"Payload erstellt: {json.dumps(payload, indent=2, ensure_ascii=False)}")
    return payload


def restore():
    """Hauptfunktion zur Wiederherstellung des Algo-Moduls."""
    # ID des Ziel-Moduls in inf_inf3
    target_id = "8cff2c5b-6f2f-4d8f-8101-f74f30c0a603"
    # ID eines funktionsfähigen Algo-Moduls (hier aus inf_inf2)
    source_id = "21723454-3c3e-4ebe-ade0-82eacb69b185"

    base_url = "https://module.gm.th-koeln.de/api"

    logger.info(f"Starte Wiederherstellung: {source_id} -> {target_id}")

    try:
        # 1. Lade Quelldaten
        source_url = f"{base_url}/modules/{source_id}"
        logger.info(f"Lade Quelldaten von {source_url}...")
        source_data = api_call(source_url)

        # 2. Mappe Daten
        payload = map_to_protocol_update(source_data)

        # 3. Update Draft
        target_url = f"{base_url}/moduleDrafts/{target_id}"
        logger.info(f"Sende Update an {target_url}...")
        headers = {"Mocogi-Version-Scheme": "v1.0s"}
        result = api_call(target_url, method="PUT", data=payload, extra_headers=headers)

        logger.info("Wiederherstellung erfolgreich abgeschlossen!")
        logger.info(f"Ergebnis: {json.dumps(result, indent=2, ensure_ascii=False)}")

    except Exception as e:
        logger.error(f"Fehler bei der Wiederherstellung: {e}")
        sys.exit(1)

if __name__ == "__main__":
    restore()
