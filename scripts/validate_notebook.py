"""Execute notebook locally with a temporary Jupyter kernel registration."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    subprocess.run(
        [sys.executable, "-m", "ipykernel", "install", "--prefix", directory, "--name", "guminet-validation"],
        check=True,
        capture_output=True,
    )
    previous = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(Path(directory) / "share/jupyter") + (
        os.pathsep + previous if previous else ""
    )
    os.environ.setdefault("MPLCONFIGDIR", str(Path(directory) / "matplotlib"))
    notebook = nbformat.read(root / "notebooks/GumiNet_Colab.ipynb", as_version=4)
    NotebookClient(
        notebook, kernel_name="guminet-validation", timeout=300, resources={"metadata": {"path": str(root)}}
    ).execute()
    if previous is None:
        os.environ.pop("JUPYTER_PATH", None)
    else:
        os.environ["JUPYTER_PATH"] = previous
    output = root / "runs/validation/GumiNet_Colab.executed.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells: {output}")
