# Changelog Workflow

Wir verwenden **Conventional Commits** und **git-cliff**, um unseren Changelog automatisch zu generieren.

## Konventionelle Commits

Bitte strukturieren Sie Ihre Commit-Nachrichten nach folgendem Schema:

```
<typ>(<bereich>): <beschreibung>

[optionaler body]

[optionaler footer]
```

### Typen  

* `feat`: Ein neues Feature
* `fix`: Ein Bugfix
* `docs`: Änderungen an der Dokumentation
* `style`: Formatierung, fehlende Semikolons, etc.
* `refactor`: Codeänderung, die weder einen Bug behebt noch ein Feature hinzufügt
* `test`: Hinzufügen von fehlenden Tests
* `chore`: Änderungen am Build-Prozess oder an Hilfsmitteln

## Automatisierung

Bei jedem Release (Push eines Tags `v*`) wird `git-cliff` ausgeführt, um die `CHANGELOG.md` zu aktualisieren und einen GitHub-Release zu erstellen.
