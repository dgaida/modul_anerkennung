"""
Skript zum Auflisten aller Pflichtveranstaltungen der Prüfungsordnung inf_inf3.
Nutzt die Mocogi-API über den MCP-Client.
"""

import asyncio
import logging
from modul_anerkennung.mcp_client import MocogiClient

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("inf3_list")

async def list_inf3_mandatory():
    """
    Halt alle Modulentwürfe ab, filtert nach inf_inf3 Pflichtmodulen und gibt sie aus.
    """
    logger.info("Starte Abfrage der inf_inf3 Pflichtmodule...")

    async with MocogiClient() as client:
        try:
            # Hole alle Entwürfe (jetzt inklusive indirect)
            drafts = await client.get_module_drafts()
            logger.info(f"Insgesamt {len(drafts)} Entwürfe geladen.")

            inf3_mandatory = []

            for draft in drafts:
                # Prüfe mandatoryPOs
                mandatory = draft.get("mandatoryPOs", [])
                if "inf_inf3" in mandatory:
                    module_info = draft.get("module", {})
                    title = module_info.get("title", "Unbekanntes Modul")
                    inf3_mandatory.append(title)

            if not inf3_mandatory:
                print("\nKeine Pflichtmodule für inf_inf3 gefunden.")
                # Debug-Hinweis falls keine gefunden wurden
                if drafts:
                    logger.debug("Beispiel-POs in den ersten 3 Drafts:")
                    for d in drafts[:3]:
                        logger.debug(f"  Modul: {d.get('module', {}).get('title')} | Mandatory: {d.get('mandatoryPOs')}")
            else:
                print(f"\nPflichtveranstaltungen in inf_inf3 ({len(inf3_mandatory)}):")
                print("-" * 50)
                for title in sorted(inf3_mandatory):
                    print(f"- {title}")
                print("-" * 50)

        except Exception as e:
            logger.error(f"Fehler bei der Abfrage: {e}")
            if "Authorization" in str(e) or "401" in str(e):
                logger.error("Bitte überprüfe den MOCOGI_API_TOKEN in deiner secrets.env.")

if __name__ == "__main__":
    asyncio.run(list_inf3_mandatory())
