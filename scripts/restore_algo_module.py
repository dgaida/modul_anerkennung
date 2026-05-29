"""
Skript zur Wiederherstellung (Restaurierung) des Moduls "Algorithmen und Datenstrukturen".
Schreibt das Modul via PUT /moduleDrafts/:id zurück in die API.
Nutzt Daten aus der PO inf_inf2 als Vorlage.
"""

import asyncio
import json
import logging
import os
import httpx
from modul_anerkennung.mcp_client import MocogiClient

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("restore_algo")

def map_to_protocol_update(source_data: dict) -> dict:
    """
    Konvertiert die API-Daten in das ModuleProtocolUpdate Format.
    Basierend auf: https://github.com/THK-ADV/modules/blob/45be4208af3ebf83ef7b587b660fd91b7e5b9df7/src/routes/my-modules/%5Bid%3Duuid%5D/%2Bpage.server.ts#L60
    """
    m = source_data.get("metadata", {})

    # Mapping auf die Struktur von ModuleProtocolUpdate
    metadata = {
        "title": "Algorithmen und Datenstrukturen",
        "abbrev": "Algo",
        "moduleType": m.get("moduleType", "module"),
        "ects": 6,
        "language": m.get("language", "de"),
        "duration": m.get("duration", 1),
        "season": m.get("season", "ss"),
        "workload": m.get("workload"),
        "status": "active",
        "location": m.get("location", "gm"),
        "participants": m.get("participants"),
        "moduleRelation": m.get("moduleRelation"),
        "moduleManagement": m.get("moduleManagement"),
        "lecturers": m.get("lecturers"),
        "assessmentMethods": {
            "mandatory": m.get("assessmentMethods", {}).get("mandatory") or []
        },
        "examiner": {
            "first": m.get("examiner", {}).get("first"),
            "second": m.get("examiner", {}).get("second")
        },
        "examPhases": m.get("examPhases") or [],
        "prerequisites": {
            "recommended": m.get("prerequisites", {}).get("recommended") or {"text": "", "modules": []},
            "required": m.get("prerequisites", {}).get("required")
        },
        "po": ["inf_inf3"],
        "taughtWith": m.get("taughtWith") or [],
        "attendanceRequirement": m.get("attendanceRequirement"),
        "assessmentPrerequisite": m.get("assessmentPrerequisite")
    }

    return {
        "metadata": metadata,
        "deContent": source_data.get("deContent") or {},
        "enContent": source_data.get("enContent") or {}
    }

async def restore():
    # ID des Ziel-Moduls in inf_inf3
    target_id = "8cff2c5b-6f2f-4d8f-8101-f74f30c0a603"

    # ID eines funktionsfähigen Algo-Moduls (hier aus inf_inf2)
    source_id = "21723454-3c3e-4ebe-ade0-82eacb69b185"

    async with MocogiClient() as client:
        logger.info(f"Lade Quelldaten von Modul {source_id}...")
        try:
            # Wir nutzen direkt httpx über den Client-Kontext um flexibel zu sein
            # (get_module_details kann bei 500 hängen bleiben)
            async with httpx.AsyncClient() as hclient:
                resp = await hclient.get(f"https://module.gm.th-koeln.de/api/modules/{source_id}")
                resp.raise_for_status()
                source_data = resp.json()
        except Exception as e:
            logger.error(f"Fehler beim Laden der Quelldaten: {e}")
            return

        # Payload vorbereiten
        payload = map_to_protocol_update(source_data)

        logger.info(f"Sende PUT Request für Modul {target_id} via update_module_draft...")

        try:
            # update_module_draft setzt automatisch die Header:
            # Authorization: Bearer $MOCOGI_API_TOKEN
            # Content-Type: application/json
            # Mocogi-Version-Scheme: v1.0s
            result = await client.update_module_draft(target_id, payload)
            logger.info("Erfolgreich wiederhergestellt!")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Fehler beim Schreiben in die API: {e}")
            logger.info("Hinweis: Falls ein 400 Bad Request auftritt, könnte dies an spezifischen Validierungen der API liegen.")

if __name__ == "__main__":
    asyncio.run(restore())
