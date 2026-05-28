# Nutzung

Das Tool bietet eine intuitive Benutzeroberfläche zur Analyse und zum Vergleich von Modulen.

## Modul-Analyse

Geben Sie den Text einer externen Modulbeschreibung ein. Das System extrahiert:  
*   **Name**: Der Titel des externen Moduls.  
*   **ECTS**: Die Anzahl der Credit Points.  
*   **Keywords**: Relevante Suchbegriffe für die interne Datenbank.  

## Suche & Vergleich

Basierend auf den extrahierten Keywords sucht das System in der Mocogi-API nach passenden internen Modulen.
Für jeden Treffer wird ein Vergleich durchgeführt:  
*   **Ähnlichkeit**: Bewertung der inhaltlichen Übereinstimmung.  
*   **Begründung**: Ein detaillierter Bericht, warum eine Anerkennung empfohlen wird oder nicht.  
*   **Status**: Ja, Nein oder Vielleicht.  

## Antrags-Erstellung

Sie können positive Vergleiche zu einer Merkliste hinzufügen. Am Ende generiert das Tool eine strukturierte Übersicht für den Prüfungsausschuss.

## Arbeiten mit Modul-Entwürfen (PO3 / inf_inf3)

Für die neue Prüfungsordnung **inf_inf3** (Informatik PO3) sind viele Module aktuell noch im Entwurfsstatus. Das Tool unterstützt den Zugriff auf diese Entwürfe (Drafts) vollumfänglich.

### Sichtbarkeit von Entwürfen
Wenn ein gültiger `MOCOGI_API_TOKEN` konfiguriert ist, lädt das System automatisch alle Entwürfe, auf die Sie Zugriff haben. Diese werden bei der Suche innerhalb der jeweiligen PO (z. B. `inf_inf3`) wie reguläre Module behandelt.

### Migration von Inhalten (PO2 zu PO3)
Um bestehende Modulbeschreibungen von der alten PO2 (`inf_inf2`) in die Entwürfe der neuen PO3 (`inf_inf3`) zu übernehmen, steht ein Migrations-Skript zur Verfügung:

```bash
# Beispiel: Migration basierend auf einer Mapping-Tabelle (Markdown)
python scripts/migrate_po_content.py mappings.md --po2 inf_inf2 --po3 inf_inf3
```

Das Skript gleicht Module anhand ihres Titels ab und kopiert die inhaltlichen Beschreibungen (`deContent`, `enContent`) vom Quell-Modul in den Ziel-Entwurf.
