"""Measured-only phase-2 report, plots, engineering checks and export."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np

from .experiment import ROOT, code_manifest, digest, read_json, write_csv, write_json


def load(path):
    with path.open(encoding='utf-8',newline='') as handle:
        return list(csv.DictReader(handle))


def build(output,cfg,smoke):
    rows=load(output/'diagnostic.csv')
    n_main=len(cfg['base_seeds'])*len(cfg['adapter_seeds'])*len(cfg['scales'])*len(cfg['main_methods'])
    n_sweep=2*len(cfg['scales'])*len(cfg['sweep_ranks'])
    n_concentrated=len(cfg['concentration_methods'])*(1 if smoke else 3)
    expected=(n_main+n_sweep+n_concentrated)*len(cfg['eval_steps'])
    results=list(output.glob('base_*/diagnostic_adapters/*/result.json'))
    records=[read_json(p) for p in results]
    checks={'expected_rows':len(rows)==expected,'expected_completed_trajectories':len(results)==n_main+n_sweep+n_concentrated,
            'zero_initialization':all(r['zero_initial_error']==0 for r in records),
            'base_frozen':all(r['base_frozen'] for r in records),
            'merge_equivalence':all(r['merge_max_error']<1e-4 for r in records),
            'source_identity':read_json(output/'diagnostic_identity.json')['code']==code_manifest(),
            'small_budget_exact':all(int(r['budget'])==cfg['budgets']['small'] for r in rows if r['regime']=='small')}
    batches={}
    for row in rows:
        key=(row['base_seed'],row['adapter_seed'],row['step'])
        batches.setdefault(key,set()).add(row['data_trace_sha256'])
    checks['paired_batch_sequences']=all(len(v)==1 for v in batches.values())
    if not all(checks.values()):
        raise AssertionError(checks)
    write_json(output/'diagnostic_verification.json',{'checks':checks,'engineering_pass':True,
               'trajectories':len(results),'evaluation_rows':len(rows),'smoke_only':smoke,
               'max_merge_error':max(r['merge_max_error'] for r in records)})
    summary=[]
    for base in cfg['base_seeds']:
        for method in cfg['main_methods']:
            for mode in cfg['scales']:
                for step in cfg['eval_steps']:
                    selected=[r for r in rows if r['regime']=='small' and int(r['base_seed'])==base
                              and r['method']==method and r['mode']==mode and int(r['step'])==step]
                    summary.append({'base_seed':base,'method':method,'mode':mode,'step':step,'adapter_seeds':len(selected),
                        'val_loss_mean':float(np.mean([float(r['val_loss']) for r in selected])),
                        'improvement_mean':float(np.mean([float(r['improvement']) for r in selected]))})
    write_csv(output/'by_base_summary.csv',summary)
    paired=[]
    for mode in cfg['scales']:
        for step in cfg['eval_steps']:
            for base in cfg['base_seeds']:
                lookup={r['method']:r['val_loss_mean'] for r in summary if r['base_seed']==base and r['mode']==mode and r['step']==step}
                for method in cfg['main_methods']:
                    paired.append({'base_seed':base,'mode':mode,'step':step,'method':method,
                                   'val_advantage_over_gradient':lookup['Gradient']-lookup[method]})
    write_csv(output/'paired_effects.csv',paired)
    fig_dir=ROOT/'figures'/output.name
    fig_dir.mkdir(parents=True,exist_ok=True)
    plot(rows,summary,cfg,fig_dir)
    lines=[f'# CarryRank 第二轮诊断：{output.name}','',
           f"工程审计通过；完成{len(results)}条轨迹、{len(rows)}条固定步数验证记录。smoke={smoke}。",
           '本轮只作机制诊断，不使用目标测试集选模，不宣称方法优势。三个基础种子各自先聚合适配种子，再比较基础种子间的配对差。','',
           f"小预算={cfg['budgets']['small']}；以下为末步验证loss的基础种子均值（±基础种子均值之间的样本std）。",
           '| 方法 | 缩放 | Val loss | 相对Gradient的配对优势 |','|---|---|---:|---|']
    last=cfg['eval_steps'][-1]
    for method in cfg['main_methods']:
        for mode in cfg['scales']:
            selected=[r for r in summary if r['method']==method and r['mode']==mode and r['step']==last]
            values=[r['val_loss_mean'] for r in selected]
            differences=[r['val_advantage_over_gradient'] for r in paired if r['method']==method and r['mode']==mode and r['step']==last]
            sd=np.std(values,ddof=1) if len(values)>1 else 0.0
            lines.append(f"| {method} | {mode} | {np.mean(values):.5f} ± {sd:.5f} | {', '.join(f'{v:.5f}' for v in differences)} |")
    lines += ['', 'QFloor正数表示：相同代理和参数预算下，保留Q并减少V后验证loss更低。该交换仍同时改变容量分布，不能等同于孤立的Q梯度因果效应。',
              '归一化使用输入二阶矩的对角代理，H_out=I，不是精确Hessian或完整K-FAC。rs_ref3=sqrt(3/r)，rank3时与旧缩放相同。',
              '原预算集中分配只用基础2026/adapter11，属于事后诊断；lr与scale的全部设置见原始表，不可只选最好的一条当主结果。',
              '不把验证指标排名、两条adapter重复或三个基础重复当成论文级确认性证据。','',
              '![Small budget](../../figures/'+output.name+'/small_budget.png)','',
              '![Capacity](../../figures/'+output.name+'/capacity.png)','',
              '![Concentration](../../figures/'+output.name+'/concentration.png)','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    published=ROOT/'results'/output.name
    if published.exists():
        raise FileExistsError('Reviewed diagnostic export already exists; do not overwrite')
    published.mkdir(parents=True)
    artifacts=[]
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.suffix in ('.csv','.json','.md','.jsonl','.txt'):
            target=published/path.relative_to(output)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            artifacts.append({'path':target.relative_to(ROOT).as_posix(),'sha256':digest(target.read_bytes()),
                              'bytes':target.stat().st_size,'source':path.relative_to(ROOT).as_posix()})
    for relative,expected_hash in read_json(output/'diagnostic_identity.json')['code'].items():
        target=published/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
        assert digest(target.read_bytes())==expected_hash
        artifacts.append({'path':target.relative_to(ROOT).as_posix(),'sha256':expected_hash,'bytes':target.stat().st_size,'source':relative})
    for path in fig_dir.glob('*'):
        artifacts.append({'path':path.relative_to(ROOT).as_posix(),'sha256':digest(path.read_bytes()),'bytes':path.stat().st_size,'source':'generated from diagnostic.csv'})
    tensors=[{'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p.read_bytes()),'bytes':p.stat().st_size} for p in output.rglob('*.pt')]
    write_json(published/'artifact-manifest.json',{'run':output.name,'files':artifacts,'local_tensors':tensors})
    for item in artifacts:
        assert digest((ROOT/item['path']).read_bytes())==item['sha256']
    print(f'DIAGNOSTIC COMPLETE: {len(results)} trajectories; {len(artifacts)} exported artifacts',flush=True)


def plot(rows,summary,cfg,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#0077BB','#009988','#EE7733','#CC3311']
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(destination/(name+'.png'),dpi=450,bbox_inches='tight')
        fig.savefig(destination/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.7))
    methods=cfg['main_methods']
    for index,mode in enumerate(cfg['scales']):
        means,errors=[],[]
        for method in methods:
            values=[r['improvement_mean'] for r in summary if r['method']==method and r['mode']==mode and r['step']==cfg['eval_steps'][-1]]
            means.append(np.mean(values));errors.append(np.std(values,ddof=1) if len(values)>1 else 0)
        ax.errorbar(np.arange(len(methods))+index*.12,means,yerr=errors,fmt='o',capsize=3,color=colors[index],label=mode)
    ax.set_xticks(np.arange(len(methods)),methods,rotation=25,ha='right')
    ax.set_ylabel('Validation improvement (nat/byte)')
    ax.set_title('Small budget: mean +/- SD of base-seed means')
    ax.legend();save(fig,'small_budget')
    fig,ax=plt.subplots(figsize=(6.5,3.5))
    for li,label in enumerate(('C','R')):
        for mi,mode in enumerate(cfg['scales']):
            selected=[r for r in rows if r['regime']=='capacity' and r['method'].startswith(label+'_') and r['mode']==mode and int(r['step'])==cfg['eval_steps'][-1]]
            selected.sort(key=lambda r:int(r['method'].split('rank')[1]))
            ax.plot([int(r['method'].split('rank')[1]) for r in selected],[float(r['improvement']) for r in selected],marker='o',color=colors[li*2+mi],label=label+' / '+mode)
    ax.set(xlabel='Rank',ylabel='Validation improvement (nat/byte)',title='Final V projection; base2026 / adapter11')
    ax.legend();save(fig,'capacity')
    fig,ax=plt.subplots(figsize=(7,3.5))
    selected=[r for r in rows if r['regime']=='concentration' and int(r['step'])==cfg['eval_steps'][-1]]
    ax.scatter(range(len(selected)),[float(r['val_loss']) for r in selected],color=colors[0])
    ax.set_xticks(range(len(selected)),[r['method']+'\n'+r['mode']+'\nlr='+r['lr'] for r in selected],rotation=35,ha='right')
    ax.set_ylabel('Validation loss (nat/byte)')
    ax.set_title('Original budget: exploratory concentration controls')
    save(fig,'concentration')
