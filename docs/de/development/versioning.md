# Versionierung

Wir verwenden **mike**, um mehrere Versionen unserer Dokumentation gleichzeitig zu verwalten und bereitzustellen.

## Deployment-Workflow

### Neue Version veröffentlichen
Wenn ein Git-Tag (z.B. `v1.0.0`) gepusht wird, erstellt die CI automatisch eine neue Version der Dokumentation:

```bash
mike deploy --push --update-aliases 1.0.0 latest
mike set-default --push latest
```

### Entwicklungs-Dokumentation
Pushes auf den `main`-Branch aktualisieren automatisch die `dev`-Version der Dokumentation.

## Lokale Vorschau von Versionen

Sie können `mike` lokal verwenden, um die versionierte Dokumentation zu testen:

```bash
# Buildet alle Versionen und startet einen Server
mike serve
```

## Versions-Switcher
Der Switcher oben rechts in der Navigationsleiste ermöglicht es Benutzern, zwischen verschiedenen Versionen (z.B. `latest`, `dev`, `v0.1.9`) zu wechseln.
