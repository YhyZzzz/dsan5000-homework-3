"""Run an explicitly requested local extraction, or refresh reports from cache."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import signal
import time

import llm_cleaning as cleaning


PROFILE=Path('reports/llm-production-profile.json')
PROGRESS=Path('reports/llm-extraction-progress.json')
PILOT_TARGETS=[
    ('AI Solution Architect (W2 Only / No C2C)',None),
    ('Senior Natural Language Processing (NLP) Engineer','Turing Talent'),
    ('Computational Scientist','Roswell Park Cancer Institute'),
    ('Quantitative Risk Analyst (Mid-level) - AML Modeling','USAA'),
    ('Consumer Insights Analyst','Earn Haus'),
]


def pilot_postings(postings):
    return [next(p for p in postings if p['payload']['title']==title and
                 (company is None or p['payload']['company_name']==company))
            for title,company in PILOT_TARGETS]


def extract(stage,workers=1):
    postings=cleaning.prepare_postings()
    profile=cleaning.model_profile()
    cleaning.save_json(PROFILE,profile)
    selected=pilot_postings(postings) if stage=='pilot' else postings
    started=time.monotonic()
    successful,cached_hits,processed,failures=0,0,0,[]
    pilot=[];stopping=False;interrupted=False
    def stop_after_active_requests(signum,frame):
        nonlocal stopping,interrupted
        stopping=True;interrupted=True
        print('Stopping after active requests finish; completed caches are preserved.',flush=True)
    previous_handlers={sig:signal.signal(sig,stop_after_active_requests) for sig in [signal.SIGINT,signal.SIGTERM]}
    executor=ThreadPoolExecutor(max_workers=workers)
    remaining=iter(selected)
    tasks={}
    def submit_next():
        posting=next(remaining,None)
        if posting is not None:
            future=executor.submit(cleaning.extract_one,posting,profile,cleaning.CACHE_DIR)
            tasks[future]=posting
    try:
        for _ in range(workers):submit_next()
        while tasks:
            done,_=wait(tasks,return_when=FIRST_COMPLETED)
            for future in done:
                posting=tasks.pop(future);processed+=1
                try:
                    result,hit=future.result()
                    if result['status']!='validated':
                        raise ValueError('Incomplete or schema-invalid generation saved for review.')
                    successful+=1;cached_hits+=int(hit)
                    row=cleaning.assemble_row(posting,result)
                    example={key:row.get(key) for key in ['job_key','job_title','salary_annual_mid','experience_years_min',
                                                        'skills','work_arrangement','visa_sponsorship','llm_postprocessing_issues']}
                    example.update(wall_seconds=result['wall_seconds'],prompt_tokens=result['prompt_eval_count'],output_tokens=result['eval_count'])
                    pilot.append(example)
                    print(f'{processed}/{len(selected)} | {posting["payload"]["title"]} | '+('cached' if hit else f'{result["wall_seconds"]:.1f}s'),flush=True)
                    if stage=='pilot':print(json.dumps(example,ensure_ascii=False),flush=True)
                except Exception as error:
                    failures.append({'job_key':posting['job_key'],'error_type':type(error).__name__})
                    print(f'{processed}/{len(selected)} | {posting["payload"]["title"]} | failed: {type(error).__name__}',flush=True)
                if len(failures)>=5:stopping=True
                cleaning.save_json(PROGRESS,{
                    'stage':stage,'model':profile['model'],'prompt_version':cleaning.PROMPT_VERSION,
                    'selected_postings':len(selected),'processed':processed,'validated':successful,'cache_hits':cached_hits,
                    'workers':workers,'base_url':cleaning.BASE_URL,'interrupted':interrupted,
                    'failed':len(failures),'failures':failures,'wall_seconds':time.monotonic()-started,
                    'updated_at_utc':datetime.now(timezone.utc).isoformat(),
                })
                if not stopping:submit_next()
    finally:
        executor.shutdown(wait=True,cancel_futures=True)
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
    history_path=Path('reports/llm-extraction-runs.json')
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    history.append({'stage':stage,'model':profile['model'],'model_digest':profile['digest'],
                    'prompt_version':cleaning.PROMPT_VERSION,'workers':workers,'postings':len(selected),
                    'validated':successful,'cache_hits':cached_hits,'failed':len(failures),'interrupted':interrupted,
                    'wall_seconds':time.monotonic()-started,'ended_at_utc':datetime.now(timezone.utc).isoformat()})
    cleaning.save_json(history_path,history)
    if stage=='pilot' and not interrupted:
        pilot_report={
            'model_profile':profile,'prompt_version':cleaning.PROMPT_VERSION,'jobs':pilot,
            'failed':len(failures),'wall_seconds':time.monotonic()-started,
        }
        cleaning.save_json('reports/llm-pilot/pilot-'+(cleaning.PROMPT_VERSION)+'.json',pilot_report)
        cleaning.save_json('reports/llm-pilot/pilot-selected.json',pilot_report)
    if failures:raise RuntimeError('Some jobs still require valid extractions. No final CSV was exported.')
    if interrupted:print('Run paused safely. Rerun the same command to resume from cache.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['pilot','extract','report'],required=True)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--base-url',default=cleaning.BASE_URL)
    args=parser.parse_args()
    if not 1<=args.workers<=8:parser.error('Use between 1 and 8 local workers.')
    if args.base_url not in {'http://127.0.0.1:11434','http://127.0.0.1:11435','http://localhost:11434'}:
        parser.error('Only the local Ollama endpoint is allowed.')
    cleaning.BASE_URL=args.base_url
    if args.stage=='report':
        report_notebook()
    else:
        extract(args.stage,args.workers)


def report_notebook():
    import tempfile
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    root=Path(__file__).resolve().parent
    path=root/'HW-3.3-cleaning-2.ipynb'
    notebook=nbformat.read(path,as_version=4)
    with tempfile.TemporaryDirectory(prefix='hw3-llm-report-') as directory:
        kernel=Path(directory)/'hw3-llm-report';kernel.mkdir()
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                                                     'display_name':'HW3 LLM Report','language':'python'}))
        manager=KernelManager(kernel_name='hw3-llm-report',kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
        executor=NotebookClient(notebook,km=manager,timeout=180,resources={'metadata':{'path':str(root)}})
        try:executor.execute(cleanup_kc=True)
        finally:nbformat.write(notebook,path)
    print('Saved cached-reporting notebook. No model requests or Git actions performed.')


if __name__=='__main__':main()
