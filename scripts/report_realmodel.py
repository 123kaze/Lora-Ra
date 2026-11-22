"""Write an experiment report using only completed real-model measurements."""
import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def main(bundle):
    decision, cfg = read(bundle/'interpretation.json'), read(bundle/'config.json')
    assert decision['complete'] and read(bundle/'checkpoint_audit.json')['complete']
    summary, intervals = rows(bundle/'summary.csv'), rows(bundle/'confidence_intervals.csv')
    outcome = ('两个任务均检出双侧谱分配的标签NLL劣势。' if decision['two_task_spectral_nll_disadvantage'] else
               '没有同时在两个任务上检出双侧谱分配的标签NLL劣势，不能确认这个推广判断。')
    lines = ['# GPT-2两任务实际补充实验', '', outcome, '',
        '本轮检验标准等宽预训练模型上的层间rank分配；不重新检验C/R、不追改第0步的B路线结论。'
        '无论方法排名如何，单个checkpoint、两个分类任务都不能支持“集中分配总是有害”。', '',
        '## 实际运行', '',
        '[预训练模型固定版本](https://huggingface.co/openai-community/gpt2/tree/607a30d783dfa663caf39e06633721c8d4cfcd7e)、'
        '[GLUE数据固定版本](https://huggingface.co/datasets/nyu-mll/glue/tree/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c)、'
        '[官方PEFT AdaLoRA接口](https://huggingface.co/docs/peft/package_reference/adalora)。实际软件版本、源代码、文件SHA另随运行保存。', '',
        f"模型参数数目：{read(bundle/'model_audit.json')['parameters']}，GPT-2固定revision。"
        'SST-2/MRPC各256条独立tune，正式验证分别872/408条，测试集未下载。'
        'SST-2训练池4096条、MRPC训练池3412条；200步按固定seed有放回抽样，共3200个样本位置，'
        '不是完整epoch。prompt末尾最多128token；只预测一个标签token。', '',
        '4方法×2任务×3个LR×独立seed0=24条调参训练；选择仅看tune最终NLL。'
        '随后锁定选择，4方法×2任务×5个新adapter/data seed=40条正式训练。'
        '全部200步，不挑最好checkpoint。5个seed只反映同一预训练checkpoint下的适配波动。', '',
        '## 正式验证结果', '',
        '均值 ± 样本SD跨5个adapter/data seed。NLL是完整词表中目标标签token的负对数概率，'
        '分类预测在预设两个标签之间取argmax。MRPC的F1为正类F1，不以NLL排名替代分类排名。', '',
        '|任务|方法|标签NLL ↓|Accuracy ↑|正类F1 ↑|', '|---|---|---:|---:|---:|']
    for row in summary:
        cells = []
        for metric in ('nll', 'accuracy', 'f1'):
            factor = 1 if metric == 'nll' else 100
            cells.append(f"{float(row[metric+'_mean'])*factor:.4f} ± {float(row[metric+'_sample_std'])*factor:.4f}"+('' if metric == 'nll' else '%'))
        lines.append(f"|{row['task']}|{row['method']}|"+'|'.join(cells)+'|')
    lines += ['', '冻结模型和不读文本的多数类预测只作参考，不参与LR选择：', '',
              '|任务|冻结模型NLL|冻结模型Accuracy|冻结模型F1|多数类Accuracy|多数类F1|',
              '|---|---:|---:|---:|---:|---:|']
    frozen = read(bundle/'frozen_baselines.json')
    for task in cfg['tasks']:
        path = next((bundle/'units').glob(f'main_{task}_*/predictions.csv'))
        labels = [int(r['label']) for r in rows(path)]
        positive, count = sum(labels), len(labels)
        majority_positive = positive > count-positive
        majority_acc = max(positive, count-positive)/count
        majority_f1 = 2*positive/(positive+count) if majority_positive else 0.
        base = frozen[task]
        lines.append(f"|{task}|{base['nll']:.4f}|{base['accuracy']*100:.2f}%|{base['f1']*100:.2f}%|{majority_acc*100:.2f}%|{majority_f1*100:.2f}%|")
    lines += ['', '## 配对差与预设判断', '',
        '差值方向按方法名顺序：Uniform−Spectral为主比较，负值表示Uniform的NLL更低；'
        'Spectral−CoverageFloor为次要比较，正值有利覆盖下限；Uniform−AdaLoRA正值有利AdaLoRA。'
        '按5个配对适配seed共享索引重采样200000次，95%边际percentile CI；不是同时置信带或后验概率。'
        '预先规定只有两个主CI上界均<0，才写“两任务均检出谱NLL劣势”。', '',
        '|任务|差值|均值|95% CI|主比较|', '|---|---|---:|---|---|']
    for row in intervals:
        lines.append(f"|{row['task']}|{row['contrast']}|{float(row['mean']):+.6f}|"
                     f"[{float(row['low']):+.6f}, {float(row['high']):+.6f}]|{row['primary']}|")
    lines += ['']
    for row in intervals:
        if row['contrast'] == 'uniform_minus_spectral':
            text = ('Uniform的标签NLL更低，区间不含0' if float(row['high']) < 0 else
                    'Spectral的标签NLL更低，区间不含0' if float(row['low']) > 0 else
                    '该差值区间包含0，未建立稳定NLL胜负')
            lines.append(f"{row['task']}：{text}。")
    if decision['classification_directions_opposite_to_nll']:
        lines += ['', '部分比较的分类指标均值方向与NLL相反；完整记录在interpretation.json。'
            '这种分歧不通过事后更换主指标解决，也不能只选择较有利的指标。']
    lines += ['', '## 分配和机制区分力', '',
        '|任务|方法|rank向量（层0→11）|HHI|非零层数|TV对Uniform|', '|---|---|---|---:|---:|---:|']
    for task in cfg['tasks']:
        allocation = read(bundle/'calibration'/task/'allocations.json')
        modules = read(bundle/'calibration'/task/'modules.json')
        names = sorted(modules, key=lambda name: modules[name]['index'])
        for method, item in allocation['methods'].items():
            ranks = [item['ranks'][name] for name in names]
            hhi = sum((r/48)**2 for r in ranks); tv = sum(abs(r-4) for r in ranks)/96
            lines.append(f"|{task}|{method}|{ranks}|{hhi:.6f}|{sum(r > 0 for r in ranks)}|{tv:.6f}|")
    equivalent = []
    for path in sorted((bundle/'units').glob('main_*/result.json')):
        item = read(path)
        if item.get('configuration_equivalent_to'):
            equivalent.append((item['task'], item['method'], item['configuration_equivalent_to']))
    if equivalent:
        entries = sorted(set(equivalent))
        lines += ['', '同配置情况：'+ '；'.join(f'{task}的{method}与{reference}相同' for task, method, reference in entries)+
            f'。共{len(equivalent)}个正式单位虽实际重训，仍不是新的分配干预；不能据其打平或微小随机差证明覆盖率机制。']
    lines += ['', 'HHI越大表示rank越集中；Uniform为1/12。TV是同形状模块参数分布的L1/(2B)。'
        '这里只含12个同形状融合QKV矩阵；所有层均非零时不能描述为“只集中到少数矩阵”。'
        'CoverageFloor同时改变了具体层的容量，因此即使更好也不隔离所有位置与优化因素。', '',
        '## 预算与学习率', '',
        '|方法族|物理可训练参数|最终有效因子参数|最终有效参数（含E）|',
        '|---|---:|---:|---:|',
        '|固定LoRA三方法|147456|147456|147456|',
        '|AdaLoRA|295008|147456|147504|', '',
        '匹配的是最终有效rank容量。AdaLoRA训练期物理参数约2倍，最终还多48个E标量（约0.0326%）；'
        '不称全程同参数、同显存或同计算。其标准SVD初始化、正交正则及rank缩放与静态rsLoRA不同，'
        '各方法在同一LR网格独立选LR，以减少优化混杂，但不是穷尽超参数搜索。', '',
        '|任务|方法|选定LR|tune NLL|', '|---|---|---:|---:|']
    for task, methods in read(bundle/'selection.json').items():
        for method, choice in methods.items():
            lines.append(f"|{task}|{method}|{choice['lr']:.0e}|{choice['tuning_nll']:.6f}|")
    boundary = [f'{task}/{method}' for task, methods in read(bundle/'selection.json').items()
                for method, choice in methods.items() if choice['lr'] in (min(cfg['learning_rates']), max(cfg['learning_rates']))]
    if boundary:
        lines += ['', '选定LR落在预定网格边界的方法：'+ '、'.join(boundary)+
            '。按协议不追加LR救结果；这些比较不是各方法经无限调优后的性能上限。']
    lines += ['', '效率实测在summary.csv中，包含训练时间、PyTorch峰值allocated及动态有效参数峰值/均值。'
        '桌面、驱动占用不在allocated中，单机计时不是专用环境下的通用性能基准。']
    profile_path = bundle/'calibration_cost_profile.json'
    if profile_path.exists():
        profile = read(profile_path)
        lines += ['', '另在全部训练完成后、无本实验训练并发时复测校准开销；无optimizer更新，'
            '不参与调参。原始数据及模型资产重新核对SHA，冻结模型验证指标也独立重载。', '',
            '|任务|校准资源复测秒数|PyTorch峰值allocated MiB|原分配复现|', '|---|---:|---:|---|']
        for row in profile['records']:
            lines.append(f"|{row['task']}|{row['seconds']:.2f}|{row['peak_cuda_allocated_bytes']/2**20:.2f}|{row['same_static_allocations']}|")
        lines += ['', f"复核资产/数据文件{profile['pinned_asset_and_data_files_rehashed']}个，冻结基线指标重载最大误差{profile['frozen_baseline_max_metric_reload_error']}。"]
    lines += ['',
        '## 正确性与边界', '',
        '全部40个正式adapter在新建适配器后独立重载，NLL/accuracy/F1误差0；'
        '初始化前向误差0，冻结基础SHA一致，真实参数和最终rank预算逐单位核对。'
        '临时FP32核对完整/末位前向和模块合并，随后恢复原精度并再核对基础SHA。'
        '原始失败smoke与成功smoke单独保留，不当成正式数据。', '',
        '校准只用训练64例，FP32前反向和FP64 CPU SVD；所有校准矩阵、前缀及精确DP可纯NumPy重算。'
        '官方验证只用于固定评价，未进入LR选择；源资产与数据切分均固定revision和SHA。'
        'MRPC的多数类准确率/F1可用逐样本标签计算，不能把高F1自动解读为强句对理解能力。', '',
        '该实验没有多个预训练种子、其他真实模型或生成任务，也没有纯浓度而位置不变的因果干预。'
        '它不能替代变宽C/R大模型验证，也不修改第0步阴性结论。GEAR未纳入下游性能比较。', '',
        '## 直接审阅', '',
        f'- [正式表](../results/{bundle.name}/main.csv)、[调参表](../results/{bundle.name}/tuning.csv)、'
        f'[全部CI](../results/{bundle.name}/confidence_intervals.csv)、[机器判断](../results/{bundle.name}/interpretation.json)。',
        f'- [主方法比较](../figures/{bundle.name}/comparison.png)、'
        f'[配对区间](../figures/{bundle.name}/paired_nll.png)、'
        f'[分配](../figures/{bundle.name}/allocation.png)。',
        f'- `python scripts/verify_realmodel_evidence.py results/{bundle.name}`：无需torch、GPU或原始runs。'
        '纯数值核验不声称无权重重载了模型；大权重和.pt只在本地保留。', '']
    report = '\n'.join(lines)
    (ROOT/'reports/realmodel_gpt2_20261122.md').write_text(report, encoding='utf-8')
    packaged = report.replace('(../results/', '(../../results/').replace('(../figures/', '(../../figures/')
    (bundle/'report.md').write_text(packaged, encoding='utf-8')
    raw = ROOT/'runs'/bundle.name
    if raw.exists():
        (raw/'report.md').write_text(packaged, encoding='utf-8')
    print(outcome)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    main(parser.parse_args().bundle.resolve())
