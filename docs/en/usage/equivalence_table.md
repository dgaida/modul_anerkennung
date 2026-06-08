# Generation of the Equivalence List (Word)

The script `scripts/create_equivalence_table_word.py` is used to generate a tabular comparison of modules from two study regulations as a Microsoft Word document (.docx). This list is an essential part of the annex for new study regulations.

## Purpose and Objective

When introducing new study regulations (PO), it must be documented which modules of the old PO correspond to which modules of the new PO. The script automates this process by:  

1. Retrieving data from existing modules (old PO) and new drafts/modules (new PO) directly from the Mocogi API.
2. Using a manual equivalence list (`data/aequivalenzliste.md`) as the basis for mapping.
3. Adding mandatory modules that are not in the list but are present in the POs.
4. Outputting the result sorted primarily by the recommended semester of the new PO in a formatted Word table.

## Prerequisites  
* **API Token**: A `MOCOGI_API_TOKEN` in `secrets.env` or `.env` is required to access the Mocogi API (especially the drafts).
* **Equivalence List**: A file named `data/aequivalenzliste.md` must exist and define the mapping.
* **Dependencies**: The `python-docx` package must be installed.

## Usage

Run the script from the project's root directory. You can specify the IDs of the old and new study regulations via parameters:

```bash
PYTHONPATH=. python3 scripts/create_equivalence_table_word.py --old-po inf_inf2 --new-po inf_inf3
```

### Parameters  
* `--old-po`: ID of the old study regulation (default: `inf_inf2`).
* `--new-po`: ID of the new study regulation (default: `inf_inf3`).

## How It Works

The script performs the following steps:  

1. **Data Acquisition**: It loads all active modules of the old PO and all drafts/modules of the new PO via the API.
2. **Mapping**: It reads the `data/aequivalenzliste.md` file. Modules are matched via their titles (case-insensitive).
3. **Completion**:
    * Mandatory modules from the old PO that have no equivalent in the list are added as a row with an empty new PO column.
    * Mandatory modules from the new PO that are not in the list are added as a row with an empty old PO column.
4. **Sorting**: The table is sorted primarily by the recommended semester of the **new** PO. If a module only exists in the old PO, its semester is used. Secondarily, sorting is alphabetical by the module title in the new PO.
5. **Formatting**: A Word document is created with optimized page margins and centered ECTS columns.

## Output

The script generates a file following the pattern `aequivalenzliste_{old_po}_{new_po}.docx` in the current working directory.
