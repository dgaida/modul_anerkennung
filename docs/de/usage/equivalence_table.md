# Erstellung der Äquivalenzliste (Word)

Das Skript `scripts/create_equivalence_table_word.py` dient dazu, eine tabellarische Gegenüberstellung von Modulen zweier Prüfungsordnungen (z. B. PO2 und PO3) als Microsoft Word-Dokument (.docx) zu generieren. Diese Liste ist ein wesentlicher Bestandteil der Anlage für neue Prüfungsordnungen.

## Sinn und Zweck
Bei der Einführung einer neuen Prüfungsordnung (PO) muss dokumentiert werden, welche Module der alten PO welchen Modulen der neuen PO entsprechen. Das Skript automatisiert diesen Prozess, indem es:
1.  Daten der bestehenden Module (PO2) und der neuen Entwürfe (PO3) direkt von der Mocogi-API abruft.
2.  Eine manuelle Äquivalenzliste (`data/aequivalenzliste.md`) als Basis für das Mapping verwendet.
3.  Module, die nicht in der Liste stehen, aber in den POs vorhanden sind, ergänzt.
4.  Das Ergebnis nach Semester und Titel sortiert in einer formatierten Word-Tabelle ausgibt.

## Voraussetzungen
- **API-Token**: Für den Zugriff auf die Mocogi-API (insbesondere die Entwürfe) ist ein `MOCOGI_API_TOKEN` in der `secrets.env` oder `.env` erforderlich.
- **Äquivalenzliste**: Eine Datei `data/aequivalenzliste.md` muss existieren und das Mapping definieren.
- **Abhängigkeiten**: Das Paket `python-docx` muss installiert sein.

## Nutzung
Führen Sie das Skript aus dem Hauptverzeichnis des Projekts aus:

```bash
PYTHONPATH=. python3 scripts/create_equivalence_table_word.py
```

## Funktionsweise
Das Skript führt folgende Schritte aus:
1.  **Datenbeschaffung**: Es lädt alle aktiven Module der PO2 (`inf_inf2`) und alle Entwürfe der PO3 (`inf_inf3`) über die API.
2.  **Mapping**: Es liest die Datei `data/aequivalenzliste.md` ein. Module werden über ihren Titel (case-insensitive) zugeordnet.
3.  **Vervollständigung**:
    -   Module aus PO2, die kein Äquivalent in der Liste haben, werden als Zeile mit leerer PO3-Spalte hinzugefügt.
    -   Module aus PO3, die nicht in der Liste stehen, werden als Zeile mit leerer PO2-Spalte hinzugefügt.
4.  **Sortierung**: Die Tabelle wird primär nach dem empfohlenen Semester und sekundär nach dem Modultitel sortiert.
5.  **Formatierung**: Es wird ein Word-Dokument erstellt, in dem die Seitenränder optimiert sind und die ECTS-Spalten zentriert werden.

## Ausgabe
Das Skript generiert die Datei `aequivalenzliste_po2_po3.docx` im aktuellen Arbeitsverzeichnis.
