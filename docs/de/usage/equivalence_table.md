# Erstellung der Äquivalenzliste (Word)

Das Skript `scripts/create_equivalence_table_word.py` dient dazu, eine tabellarische Gegenüberstellung von Modulen zweier Prüfungsordnungen als Microsoft Word-Dokument (.docx) zu generieren. Diese Liste ist ein wesentlicher Bestandteil der Anlage für neue Prüfungsordnungen.

## Sinn und Zweck

Bei der Einführung einer neuen Prüfungsordnung (PO) muss dokumentiert werden, welche Module der alten PO welchen Modulen der neuen PO entsprechen. Das Skript automatisiert diesen Prozess, indem es:  

1. Daten der bestehenden Module (alte PO) und der neuen Entwürfe (neue PO) direkt von der Mocogi-API abruft.
1. Eine manuelle Äquivalenzliste (`data/aequivalenzliste.md`) als Basis für das Mapping verwendet.
1. Pflichtmodule, die nicht in der Liste stehen, aber in den POs vorhanden sind, ergänzt.
1. Das Ergebnis primär nach dem empfohlenen Semester der neuen PO sortiert in einer formatierten Word-Tabelle ausgibt.

## Voraussetzungen  
* **API-Token**: Für den Zugriff auf die Mocogi-API (insbesondere die Entwürfe) ist ein `MOCOGI_API_TOKEN` in der `secrets.env` oder `.env` erforderlich.
* **Äquivalenzliste**: Eine Datei `data/aequivalenzliste.md` muss existieren und das Mapping definieren.
* **Abhängigkeiten**: Das Paket `python-docx` muss installiert sein.

## Nutzung

Führen Sie das Skript aus dem Hauptverzeichnis des Projekts aus. Sie können die IDs der alten und neuen Prüfungsordnung über Parameter angeben:

```bash
PYTHONPATH=. python3 scripts/create_equivalence_table_word.py --old-po inf_inf2 --new-po inf_inf3
```

### Parameter  
* `--old-po`: ID der alten Prüfungsordnung (Standard: `inf_inf2`).
* `--new-po`: ID der neuen Prüfungsordnung (Standard: `inf_inf3`).

## Funktionsweise

Das Skript führt folgende Schritte aus:  

1. **Datenbeschaffung**: Es lädt alle aktiven Module der alten PO und alle Entwürfe/Module der neuen PO über die API.
1. **Mapping**: Es liest die Datei `data/aequivalenzliste.md` ein. Module werden über ihren Titel (case-insensitive) zugeordnet.
1. **Vervollständigung**:
    * Pflichtmodule aus der alten PO, die kein Äquivalent in der Liste haben, werden als Zeile mit leerer neuer PO-Spalte hinzugefügt.
    * Pflichtmodule aus der neuen PO, die nicht in der Liste stehen, werden als Zeile mit leerer alter PO-Spalte hinzugefügt.
1. **Sortierung**: Die Tabelle wird primär nach dem empfohlenen Semester der **neuen** PO sortiert. Falls ein Modul nur in der alten PO existiert, wird dessen Semester verwendet. Sekundär erfolgt die Sortierung alphabetisch nach dem Titel des Moduls in der neuen PO.
1. **Formatierung**: Es wird ein Word-Dokument erstellt, in dem die Seitenränder optimiert sind und die ECTS-Spalten zentriert werden.

## Ausgabe

Das Skript generiert eine Datei nach dem Muster `aequivalenzliste_{old_po}_{new_po}.docx` im aktuellen Arbeitsverzeichnis.
