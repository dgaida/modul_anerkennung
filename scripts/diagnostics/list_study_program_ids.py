#!/usr/bin/env python3
"""
Skript zum Auflisten aller verfügbaren Studiengang-IDs (POs) aus der Mocogi-API.
"""

import asyncio
import httpx
import os
import logging

API_BASE_URL = "https://module.gm.th-koeln.de/api"

logging.basicConfig(level=logging.INFO, format="%(message)s")

def get_headers():
    headers = {}
    token = os.getenv("MOCOGI_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

async def main():
    async with httpx.AsyncClient() as client:
        print("Rufe Studiengänge ab...")
        resp = await client.get(f"{API_BASE_URL}/studyPrograms", headers=get_headers())

        if resp.status_code == 200:
            progs = resp.json()
            print(f"Gefunden: {len(progs)} Einträge\n")

            print(f"{'PO-ID':<15} | {'Version':<7} | {'Name'}")
            print("-" * 50)

            # Sortieren nach ID
            sorted_progs = sorted(progs, key=lambda x: x.get("po", {}).get("id", ""))

            for p in sorted_progs:
                po = p.get("po", {})
                po_id = po.get("id", "N/A")
                version = po.get("version", "?")
                label = p.get("deLabel", "N/A")
                print(f"{po_id:<15} | {version:<7} | {label}")
        else:
            print(f"Fehler: {resp.status_code}")

if __name__ == "__main__":
    asyncio.run(main())
