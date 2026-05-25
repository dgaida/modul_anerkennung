#!/usr/bin/env python3
"""
Skript zur Überprüfung der Modulzahlen für spezifische Prüfungsordnungen (PO).
Prüft sowohl aktive als auch inaktive Module.
"""

import asyncio
import httpx
import os
import argparse
import logging

# API-Konfiguration
API_BASE_URL = "https://module.gm.th-koeln.de/api"

# Logging konfigurieren
# Wir unterdrücken httpx Logs für eine sauberere Ausgabe
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("verify_po")

def get_headers():
    """Erstellt die HTTP-Header für die API-Anfragen."""
    headers = {}
    token = os.getenv("MOCOGI_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

async def check_po(client, po_id):
    """Prüft die Modulzahlen für eine bestimmte PO ID."""
    print(f"\n--- Untersuchung PO: {po_id} ---")

    # 1. Aktive Module (active=true)
    params_active = {"select": "metadata", "active": "true", "po": po_id}
    resp_active = await client.get(f"{API_BASE_URL}/modules", params=params_active, headers=get_headers())

    # 2. Inaktive Module (active=false)
    params_inactive = {"select": "metadata", "active": "false", "po": po_id}
    resp_inactive = await client.get(f"{API_BASE_URL}/modules", params=params_inactive, headers=get_headers())

    def get_count(resp):
        if resp.status_code == 200:
            try:
                return len(resp.json())
            except:
                return "JSON_ERROR"
        return 0 if resp.status_code == 404 else f"HTTP_{resp.status_code}"

    count_active = get_count(resp_active)
    count_inactive = get_count(resp_inactive)

    print(f"  Aktive Module (active=true):   {count_active}")
    print(f"  Inaktive Module (active=false): {count_inactive}")

    return count_active

async def main():
    parser = argparse.ArgumentParser(description="Prüft Modulzahlen für PO-IDs.")
    parser.add_argument("ids", nargs="*", default=["inf_inf1", "inf_inf2", "inf_inf3", "inf_inf4"],
                        help="Liste von PO-IDs (z.B. inf_inf2 inf_inf3)")
    args = parser.parse_args()

    async with httpx.AsyncClient() as client:
        # Erst PO Existenz prüfen
        try:
            resp = await client.get(f"{API_BASE_URL}/studyPrograms", headers=get_headers())
            resp.raise_for_status()
        except Exception as e:
            logger.error(f"Fehler beim Laden der Studiengänge: {e}")
            return

        progs = resp.json()
        program_ids = {p.get("po", {}).get("id") for p in progs if p.get("po", {}).get("id")}

        for po_id in args.ids:
            if po_id not in program_ids:
                print(f"\n[!] PO ID '{po_id}' wurde nicht in der Liste der Studiengänge gefunden.")
                # Suche nach ähnlichen IDs
                similar = sorted([pid for pid in program_ids if pid and po_id[:7] in pid])
                if similar:
                    print(f"    Ähnliche IDs im System: {similar}")

            await check_po(client, po_id)

if __name__ == "__main__":
    asyncio.run(main())
