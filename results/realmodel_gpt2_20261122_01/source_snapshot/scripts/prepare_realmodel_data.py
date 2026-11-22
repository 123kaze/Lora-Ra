"""Use the existing CPU Arrow environment to make fixed train/tune splits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT/'configs/realmodel_gpt2.json').read_text(encoding='utf-8'))
    raw, destination = ROOT/cfg['assets']/'dataset', ROOT/cfg['data']
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for task in cfg['tasks']:
        train = pq.read_table(raw/task/'train-00000-of-00001.parquet').to_pylist()
        validation = pq.read_table(raw/task/'validation-00000-of-00001.parquet').to_pylist()
        order = np.random.default_rng(cfg['split_seed']).permutation(len(train)).tolist()
        tune_ids, train_ids = order[:cfg['tune_examples']], order[cfg['tune_examples']:]
        if task == 'sst2':
            train_ids = train_ids[:cfg['sst2_train_examples']]
        assert not set(tune_ids)&set(train_ids)
        for split, data in [('train', [train[i] for i in train_ids]), ('tune', [train[i] for i in tune_ids]), ('validation', validation)]:
            path = destination/f'{task}_{split}.jsonl'
            path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in data), encoding='utf-8')
            records.append({'task': task, 'split': split, 'examples': len(data),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'path': path.relative_to(ROOT).as_posix(),
                'positive_labels': sum(int(r['label']) for r in data)})
        (destination/f'{task}_split_ids.json').write_text(json.dumps({'train_row_indices': train_ids, 'tune_row_indices': tune_ids}), encoding='utf-8')
    value = {'split_seed': cfg['split_seed'], 'test_read': False, 'records': records,
             'source_assets_sha256': hashlib.sha256((ROOT/cfg['assets']/'assets.json').read_bytes()).hexdigest()}
    (destination/'manifest.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
