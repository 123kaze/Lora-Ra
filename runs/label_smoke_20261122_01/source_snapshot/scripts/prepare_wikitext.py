"""Pinned WikiText-2 raw conversion; can run in a separate Python without torch."""
import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'
EXPECTED = {
    'train': ('e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7', 36718),
    'validation': ('204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c', 3760),
}


def main():
    raw = ROOT / 'data/raw/wikitext-2-raw-v1'
    processed = ROOT / 'data/processed/wikitext-2-raw-v1'
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    records = []
    for split, (expected, rows) in EXPECTED.items():
        name = f'{split}-00000-of-00001.parquet'
        url = f'https://huggingface.co/datasets/Salesforce/wikitext/resolve/{REVISION}/wikitext-2-raw-v1/{name}'
        path = raw / name
        if not path.exists():
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'CarryRank-Research/0.3'}), timeout=60) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Download hash mismatch: ' + name)
            path.write_bytes(content)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError('Cached parquet hash mismatch: ' + name)
        lines = pq.read_table(path).column('text').to_pylist()
        if len(lines) != rows or not all(isinstance(line, str) for line in lines):
            raise ValueError('Unexpected schema or row count: ' + name)
        encoded = ''.join(lines).encode('utf-8')
        destination = processed / f'{split}.bin'
        destination.write_bytes(encoded)
        records.append({'split': split, 'url': url, 'rows': rows,
                        'parquet_sha256': expected, 'parquet_bytes': len(content),
                        'bin_path': destination.relative_to(ROOT).as_posix(),
                        'bin_bytes': len(encoded), 'bin_sha256': hashlib.sha256(encoded).hexdigest()})
    manifest = {'dataset': 'Salesforce/wikitext', 'subset': 'wikitext-2-raw-v1',
                'revision': REVISION, 'license_tags': ['cc-by-sa-3.0', 'gfdl'],
                'transform': "UTF8(''.join(text_rows_in_original_order)); no extra separator or normalization",
                'test_downloaded_or_used': False, 'splits': records}
    (processed / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
