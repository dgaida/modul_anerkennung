# Usage

The tool provides an intuitive user interface for analyzing and comparing modules.

## Module Analysis

Enter the text of an external module description. The system extracts:
*   **Name**: The title of the external module.
*   **ECTS**: The number of credit points.
*   **Keywords**: Relevant search terms for the internal database.

## Search & Comparison

Based on the extracted keywords, the system searches the Mocogi API for matching internal modules.
For each hit, a comparison is performed:
*   **Similarity**: Assessment of content agreement.
*   **Justification**: A detailed report on why recognition is recommended or not.
*   **Status**: Yes, No, or Maybe.

## Application Creation

You can add positive comparisons to a watchlist. At the end, the tool generates a structured overview for the examination board.

## Working with Module Drafts (PO3 / inf_inf3)

For the new examination regulations **inf_inf3** (Computer Science PO3), many modules are currently still in draft status. The tool fully supports access to these drafts.

### Visibility of Drafts
If a valid `MOCOGI_API_TOKEN` is configured, the system automatically loads all drafts to which you have access. These are treated like regular modules during the search within the respective PO (e.g., `inf_inf3`).

### Content Migration (PO2 to PO3)
A migration script is available to transfer existing module descriptions from the old PO2 (`inf_inf2`) to the drafts of the new PO3 (`inf_inf3`):

```bash
# Example: Migration based on a mapping table (Markdown)
python scripts/migrate_po_content.py mappings.md --po2 inf_inf2 --po3 inf_inf3
```

The script matches modules based on their title and copies the content descriptions (`deContent`, `enContent`) from the source module to the target draft.
