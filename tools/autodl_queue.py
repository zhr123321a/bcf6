"""Persistent, GPU-isolated BCF6 multi-scene queue for Linux."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse, csv, fcntl, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback

def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
def read(p): return json.loads(Path(p).read_text())
def save(p, v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def ident(pid):
    try:
        p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split()
        return None if v[0]=='Z' else {'pid':pid,'start_ticks':int(v[19])}
    except (OSError,ValueError): return None
def cards():
    def query(flag):
        x=subprocess.check_output(['nvidia-smi',flag,'--format=csv,noheader,nounits'],text=True,timeout=20)
        return [[v.strip() for v in row] for row in csv.reader(x.splitlines()) if row]
    result={int(i):{'index':int(i),'uuid':u,'free_mib':int(f),'processes':[]} for i,u,f in query('--query-gpu=index,uuid,memory.free')}
    lookup={x['uuid']:x for x in result.values()}
    for u,p,n,m in query('--query-compute-apps=gpu_uuid,pid,process_name,used_memory'):
        if u in lookup: lookup[u]['processes'].append({'pid':int(p),'name':n,'memory_mib':m})
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args();cfg=read(a.config)
    root=Path(cfg['run']);data=Path(cfg['data']);source=Path(cfg['source']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'supervisor.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(source));from bcf6.train import source_identity
    src=source_identity();assert src['combined_sha256']==cfg['expected_source_sha256'],src
    assert len(cfg['scenes'])==len(set(cfg['scenes'])) and cfg['gpus'] and len(cfg['gpus'])<=6
    statefile=root/'status.json';state=read(statefile) if statefile.exists() else {'start_bjt':now(),'state':'running','jobs':[],'source':src,'config':cfg}
    if not state['jobs']:
        for p in cfg['profiles']:
            state['jobs'].append({'id':'gate_'+p,'scene':cfg['scenes'][0],'profile':p,'gate':True,'state':'pending','attempt':0})
        for scene in cfg['scenes']:
            for p in cfg['profiles']:state['jobs'].append({'id':scene+'_'+p,'scene':scene,'profile':p,'gate':False,'state':'pending','attempt':0})
    alive={};prepared={};terminal={'complete','failed','skipped_gate'};save(statefile,state)
    while True:
        try:
            for j in state['jobs']:
                if j['state']!='running':continue
                proc=alive.get(j['id']);returncode=proc.poll() if proc else None;live=ident(j['pid'])==j['process_identity']
                if returncode is None and live:continue
                if proc and returncode is None:continue
                cp=Path(j['out'])/'complete.json';ok=False
                if cp.exists():
                    d=read(cp);ok=d.get('status')=='complete' and d.get('replay',{}).get('continuation_allowed') is True and d.get('target_sha256')==j['target_sha256']
                if returncode not in (None,0):ok=False
                j.update(state='complete' if ok else 'failed',return_code=returncode,end_bjt=now(),failure='' if ok else 'Trainer exited without validated saved result; see log')
                alive.pop(j['id'],None)
            gates={j['profile']:j['state'] for j in state['jobs'] if j['gate']}
            for j in state['jobs']:
                if j['state']=='pending' and not j['gate'] and gates[j['profile']] in ('failed','skipped_gate'):
                    j.update(state='skipped_gate',end_bjt=now(),failure='Matching profile gate failed')
            used={j['gpu'] for j in state['jobs'] if j['state']=='running'};snapshot=cards();state['gpu_snapshot']=snapshot
            free_disk=shutil.disk_usage(root).free;state['disk_free_bytes']=free_disk
            if free_disk < cfg.get('disk_reserve_bytes',8*1024**3): state['wait_reason']='disk reserve'
            else:
                state['wait_reason']='';available=[g for g in cfg['gpus'] if g not in used and g in snapshot and not snapshot[g]['processes'] and snapshot[g]['free_mib']>=cfg.get('required_free_mib',8192)]
                for j in state['jobs']:
                    if not available:break
                    if j['state']!='pending' or (not j['gate'] and gates[j['profile']]!='complete'):continue
                    scene=j['scene'];idir=data/'materials'/scene;target=idir/'target.f32';original=idir/'identity.json'
                    if not original.is_file() or not target.is_file():continue
                    ref=read(original);expected=int(ref['width'])*int(ref['height'])*9*4
                    if target.stat().st_size!=expected:continue
                    if scene not in prepared:
                        if sha(target)!=ref['target_sha256']:
                            j.update(state='failed',end_bjt=now(),failure='Target SHA mismatch');continue
                        new=dict(ref,target_f32=str(target));identity=root/'identities'/(scene+'.json');save(identity,new);prepared[scene]=(identity,ref['target_sha256'])
                    identity,digest=prepared[scene];g=available.pop(0);out=root/('gates' if j['gate'] else 'results')/j['profile']/scene;out.mkdir(parents=True,exist_ok=True)
                    steps=['3','8','2'] if j['gate'] else ['5000','200000','1000']
                    argv=[sys.executable,'-m','bcf6.train','--identity',str(identity),'--out',str(out),'--profile',j['profile'],'--outputs','8','--warmup-steps',steps[0],'--block-steps',steps[1],'--finetune-steps',steps[2],'--batch-side','512','--seed','1','--device','cuda','--threads','2','--block-backend','eager','--checkpoint-every','1000','--log-every','2' if j['gate'] else '100']
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=snapshot[g]['uuid'],PYTHONPATH=str(source),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TMPDIR=str(root/'tmp'),TORCHINDUCTOR_CACHE_DIR=str(root/'cache'));(root/'tmp').mkdir(exist_ok=True)
                    log=out/'scheduler.log';j.update(state='claimed',gpu=g,gpu_uuid=snapshot[g]['uuid'],out=str(out),argv=argv,attempt=j['attempt']+1,start_bjt=now(),log=str(log),target_sha256=digest);save(statefile,state)
                    with log.open('ab') as f:proc=subprocess.Popen(argv,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    j.update(state='running',pid=proc.pid,process_identity=ident(proc.pid));alive[j['id']]=proc;save(statefile,state)
            state['counts']={x:sum(j['state']==x for j in state['jobs']) for x in ('pending','running','complete','failed','skipped_gate')};state['updated_bjt']=now()
            if all(j['state'] in terminal for j in state['jobs']):
                state['state']='complete' if not any(j['state']!='complete' for j in state['jobs']) else 'finished_with_failures';state['end_bjt']=now();save(statefile,state)
                formals=[j for j in state['jobs'] if not j['gate']];lines=['# BCF6 场景结果\n\n| 场景 | 档位 | 状态 | RGB9 PSNR / dB | 数值载荷 bpp |\n|---|---|---|---:|---:|\n']
                for j in formals:
                    d=read(Path(j['out'])/'complete.json') if j['state']=='complete' else None
                    lines.append('| %s | %s | %s | %s | %s |\n'%(j['scene'],j['profile'],j['state'],('%.6f'%d['quality']['rgb9_psnr'] if d else '—'),('%.6f'%d['payload_bpp'] if d else '—')))
                (root/'PER_SCENE.md').write_text(''.join(lines));save(root/'report_complete.json',{'completed':sum(j['state']=='complete' for j in formals),'total':len(formals),'end_bjt':now()});return
            save(statefile,state);time.sleep(5)
        except Exception:
            with (root/'supervisor_errors.log').open('a') as f:f.write(now()+'\n'+traceback.format_exc()+'\n')
            time.sleep(10)
if __name__=='__main__':main()
