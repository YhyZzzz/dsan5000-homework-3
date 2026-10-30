"""Execute and save Part 1, offline by default. No Git operations are performed."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly enable real SerpApi requests.")
    parser.add_argument("--stage", choices=["trial", "second", "all"], default="all")
    parser.add_argument("--keys-file", type=Path, help="JSON key file outside the repository.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    path = root / "HW-3.1-crawl.ipynb"
    os.environ["HW3_RUN_LIVE"] = "1" if args.live else "0"
    os.environ["HW3_STAGE"] = args.stage
    if args.keys_file:
        os.environ["HW3_KEYS_FILE"] = str(args.keys_file.expanduser().resolve())
    notebook = nbformat.read(path, as_version=4)
    # Use a temporary kernel spec so no global Jupyter configuration is changed.
    with tempfile.TemporaryDirectory(prefix="hw3-kernel-") as directory:
        kernel_folder = Path(directory) / "hw3"
        kernel_folder.mkdir()
        (kernel_folder / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Homework 3 Python", "language": "python",
        }), encoding="utf-8")
        manager = KernelManager(
            kernel_name="hw3",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
        )
        executor = NotebookClient(
            notebook,
            km=manager,
            timeout=None,  # The paced full crawl takes about two hours.
            resources={"metadata": {"path": str(root)}},
        )
        try:
            executor.execute(cleanup_kc=True)
        finally:
            # Preserve actual outputs even if collection is interrupted or fails.
            nbformat.write(notebook, path)
    print(f"Saved executed notebook: {path.name}")
    print("Inspect the collection audit before committing. No Git actions were performed.")


if __name__ == "__main__":
    main()
