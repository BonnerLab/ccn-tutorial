#!/.venv/bin/python

import hashlib
from pathlib import Path

import nbformat
from nbformat import NotebookNode


def add_content_adressed_cell_ids(notebook: NotebookNode) -> NotebookNode:
    ids = {}

    for i_cell, cell in enumerate(notebook["cells"]):
        if cell["id"][:4] != "cell":
            id_ = hashlib.sha1(cell["source"].encode()).hexdigest()
            if id_ not in ids:
                ids[id_] = 0
            else:
                ids[id_] += 1

            notebook["cells"][i_cell]["id"] = f"{id_}-{ids[id_]}"[-64:]
    return notebook


for filepath in Path("notebooks").rglob("*.ipynb"):
    notebook = nbformat.read(filepath, as_version=4)
    notebook = add_content_adressed_cell_ids(notebook)
    notebook = nbformat.write(notebook, filepath, version=4)
