"""Einstiegspunkt für das Tool."""

import os
import logging
from modul_anerkennung.gui import launch_gui, CSS

# Logging initialisieren
# Wir prüfen DEBUG oder LOG_LEVEL
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
if os.environ.get("DEBUG", "false").lower() == "true":
    log_level = logging.DEBUG
else:
    log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Port für Render.com oder lokale Entwicklung
    port = int(os.environ.get("PORT", 7860))
    logger.debug(f"Konfigurierter Port: {port}")

    # GUI initialisieren
    logger.info("Initialisiere GUI...")
    demo = launch_gui()

    # GUI starten
    logger.info(f"Starte Gradio-Server auf http://0.0.0.0:{port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CSS,
        share=False
    )
