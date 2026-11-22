"""Download only pinned public model files and GLUE train/validation assets."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT/'configs/realmodel_gpt2.json').read_text(encoding='utf-8'))
    destination = ROOT/cfg['assets']
    files = []
    model_names = ['config.json', 'merges.txt', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'model.safetensors']
    requested = [('model', cfg['model_id'], cfg['model_revision'], name) for name in model_names]
    requested += [('dataset', cfg['dataset_id'], cfg['dataset_revision'], f'{task}/{split}-00000-of-00001.parquet')
                  for task in cfg['tasks'] for split in ('train', 'validation')]
    for kind, repo, revision, name in requested:
        path = Path(hf_hub_download(repo_id=repo, repo_type=kind, revision=revision, filename=name,
                                   local_dir=destination/kind, token=False))
        files.append({'repo': repo, 'revision': revision, 'kind': kind, 'filename': name,
                      'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(f'{kind}/{name}: {path.stat().st_size} bytes verified', flush=True)
    (destination/'assets.json').write_text(json.dumps({'files': files, 'test_downloaded': False}, indent=2), encoding='utf-8')
    print('Pinned assets complete', flush=True)


if __name__ == '__main__':
    main()
