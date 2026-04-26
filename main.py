"""Einstiegspunkt für das Tool."""

import os
from modul_anerkennung.gui import launch_gui, CSS

if __name__ == "__main__":
    # Port für Render.com oder lokale Entwicklung
    port = int(os.environ.get("PORT", 7860))

    # GUI initialisieren
    demo = launch_gui()

    # GUI starten
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CSS,
        share=False
    )
