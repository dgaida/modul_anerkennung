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
