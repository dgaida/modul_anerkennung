# Creation of the Equivalence Table (Word)

The script `scripts/create_equivalence_table_word.py` is used to generate a tabular comparison of modules from two study regulations (e.g., PO2 and PO3) as a Microsoft Word document (.docx). This list is an essential part of the annex for new study regulations.

## Purpose and Objective
When introducing new study regulations (PO), it must be documented which modules of the old PO correspond to which modules of the new PO. The script automates this process by:  
1.  Retrieving data from existing modules (PO2) and new drafts (PO3) directly from the Mocogi API.  
2.  Using a manual equivalence list (`data/aequivalenzliste.md`) as the basis for mapping.  
3.  Adding modules that are not in the list but are present in the POs.  
4.  Outputting the result sorted by semester and title in a formatted Word table.  

## Prerequisites  
- **API Token**: A `MOCOGI_API_TOKEN` in `secrets.env` or `.env` is required to access the Mocogi API (especially the drafts).  
- **Equivalence List**: A file named `data/aequivalenzliste.md` must exist and define the mapping.  
- **Dependencies**: The `python-docx` package must be installed.  

## Usage
Run the script from the project's root directory:

```bash
PYTHONPATH=. python3 scripts/create_equivalence_table_word.py
```

## How It Works
The script performs the following steps:  
1.  **Data Acquisition**: It loads all active modules of PO2 (`inf_inf2`) and all drafts of PO3 (`inf_inf3`) via the API.  
2.  **Mapping**: It reads the `data/aequivalenzliste.md` file. Modules are matched via their titles (case-insensitive).  
3.  **Completion**:  
    -   Modules from PO2 that have no equivalent in the list are added as a row with an empty PO3 column.  
    -   Modules from PO3 that are not in the list are added as a row with an empty PO2 column.  
4.  **Sorting**: The table is sorted primarily by the recommended semester and secondarily by the module title.  
5.  **Formatting**: A Word document is created with optimized page margins and centered ECTS columns.  

## Output
The script generates the file `aequivalenzliste_po2_po3.docx` in the current working directory.
