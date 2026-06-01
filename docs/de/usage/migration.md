# Migration von Modulinhalten via Liste

Das Skript `scripts/migrate_from_list.py` ermöglicht die automatisierte Übernahme von Modulinhalten (Deutsch und Englisch) aus einer bestehenden Prüfungsordnung (z. B. PO2) in die Entwürfe einer neuen Prüfungsordnung (z. B. PO3).

## Ziel und Zweck
Das Ziel ist es, die mühsame manuelle Übertragung von Lehrinhalten zu vermeiden, wenn Module in einer neuen PO identisch oder sehr ähnlich bleiben. Das Skript stellt sicher, dass die inhaltlichen Beschreibungen konsistent übernommen werden, während die Identität und spezifische Metadaten (wie IDs) des Ziel-Entwurfs erhalten bleiben.

## Voraussetzungen und Abhängigkeiten
Das Skript ist als Standalone-Werkzeug konzipiert, nutzt jedoch Logik aus anderen Skripten:
- **Abhängigkeit**: `scripts/standalone_restore_algo.py` (muss im selben Verzeichnis liegen, da Funktionen daraus importiert werden).
- **Eingabedatei**: `data/aequivalenzliste.md` (enthält das Mapping).

## Vorbereitung der Äquivalenzliste
Die Liste muss als Markdown-Tabelle in `data/aequivalenzliste.md` vorliegen. Das Format ist wie folgt:

```markdown
| Modul in PO2 | Modul in PO3 | Semester |
| :--- | :--- | :--- |
| Algorithmik | Algorithmen und Datenstrukturen | 2 |
| Mathematik 1 | Mathe 1 | 1 |
```

- **Spalte 1**: Der exakte Titel des Quell-Moduls in der PO `inf_inf2`.
- **Spalte 2**: Der exakte Titel des Ziel-Entwurfs in der PO `inf_inf3`.
- **Spalte 3**: Das empfohlene Semester für das Ziel-Modul.

## Authentifizierung (API-Token)
Für den Schreibzugriff auf die Mocogi-API wird ein gültiger `MOCOGI_API_TOKEN` benötigt.

### Token erhalten
1. Melden Sie sich im Browser bei [module.gm.th-koeln.de](https://module.gm.th-koeln.de) an.
2. Öffnen Sie die Entwicklertools (F12) -> Tab "Application" (Chromium) oder "Speicher" (Firefox) oder "Netzwerk".
3. Suchen Sie unter "Cookies" nach dem Eintrag für die Domain oder schauen Sie in die Request-Header eines API-Aufrufs.
4. Kopieren Sie den Wert des Tokens (meist `access_token` oder ähnlich).

### Token speichern
Erstellen Sie im Wurzelverzeichnis des Projekts eine Datei namens `secrets.env` (falls nicht vorhanden) und fügen Sie den Token ein:

```env
MOCOGI_API_TOKEN=ihr_kopierter_token_hier
```

## Skript starten
Führen Sie das Skript aus dem Hauptverzeichnis des Projekts aus:

```bash
PYTHONPATH=. python3 scripts/migrate_from_list.py
```

Das Skript gibt detaillierte Logs über den Fortschritt und eventuelle Fehler aus.
