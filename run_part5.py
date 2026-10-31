"""Generate fictional applications, render Quarto artifacts, or run cached reporting."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import resume_pipeline as pipeline


def report():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    root=Path(__file__).resolve().parent;path=root/'HW-3.5-optional-bonus-resume-creator.ipynb'
    notebook=nbformat.read(path,as_version=4)
    with tempfile.TemporaryDirectory(prefix='hw3-bonus-') as directory:
        kernel=Path(directory)/'bonus-report';kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
          'display_name':'HW3 Bonus','language':'python'}))
        manager=KernelManager(kernel_name='bonus-report',kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
        executor=NotebookClient(notebook,km=manager,timeout=180,resources={'metadata':{'path':str(root)}})
        try:executor.execute(cleanup_kc=True)
        finally:nbformat.write(notebook,path)
    print('Saved executed bonus notebook; no model calls during reporting.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['generate','render','report'],required=True)
    parser.add_argument('--base-url',default=pipeline.BASE_URL)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.base_url not in ['http://127.0.0.1:11434','http://127.0.0.1:11435']:parser.error('Local model endpoints only.')
    if not 1<=args.workers<=4:parser.error('Use 1–4 workers.')
    pipeline.BASE_URL=args.base_url
    if args.stage=='generate':pipeline.generate_all(args.workers)
    elif args.stage=='render':
        jobs=json.loads(Path('reports/bonus/selected-jobs.json').read_text())
        profile=json.loads(Path('reports/bonus/model-profile.json').read_text())
        rows=pipeline.render_documents(jobs,profile);print('Rendered',len(rows),'fictional resume/cover-letter pairs.')
    else:report()

if __name__=='__main__':main()
