"""Execute the deterministic cleaning notebook and save its outputs; no Git actions."""

import json
from pathlib import Path
import sys
import tempfile

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parent
    path = root / 'HW-3.2-cleaning-1.ipynb'
    notebook = nbformat.read(path, as_version=4)
    with tempfile.TemporaryDirectory(prefix='hw3-cleaning-kernel-') as directory:
        kernel = Path(directory) / 'hw3-cleaning'
        kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Homework 3 Cleaning', 'language': 'python',
        }), encoding='utf-8')
        manager = KernelManager(
            kernel_name='hw3-cleaning',
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
        )
        executor = NotebookClient(notebook, km=manager, timeout=180,
                                  resources={'metadata': {'path':str(root)}})
        try:
            executor.execute(cleanup_kc=True)
        finally:
            nbformat.write(notebook,path)
    print('Saved executed Part 2 notebook, processed CSV, and cleaning reports.')
    print('No API calls, Git commits, or pushes were performed.')


if __name__ == '__main__':
    main()
