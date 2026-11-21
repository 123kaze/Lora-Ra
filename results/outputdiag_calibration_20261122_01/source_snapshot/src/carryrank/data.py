"""Byte LM data with contiguous, nonoverlapping source/target splits."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch


def digest(data: bytes):
    return hashlib.sha256(data).hexdigest()


def prepare(root: Path, config: dict):
    raw = root / "data/raw/tinyshakespeare.txt"
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        request = urllib.request.Request(config["url"], headers={"User-Agent": "CarryRank-Pilot/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        if len(content) < 100000:
            raise ValueError("Unexpectedly small corpus download")
        raw.write_bytes(content)
    content = raw.read_bytes()
    names = ["source_train", "source_val", "target_train", "target_val", "target_test"]
    boundaries = [int(len(content) * f) for f in config["fractions"]]
    if len(boundaries) != 6 or boundaries[0] != 0 or boundaries[-1] != len(content):
        raise ValueError("Invalid split fractions")
    splits, records = {}, []
    for name, start, end in zip(names, boundaries, boundaries[1:]):
        if end <= start:
            raise ValueError("Empty split")
        value = content[start:end]
        splits[name] = torch.from_numpy(np.frombuffer(value, dtype=np.uint8).copy()).long()
        records.append({"name": name, "start": start, "end": end, "bytes": len(value), "sha256": digest(value)})
    manifest = {"url": config["url"], "raw_sha256": digest(content), "raw_bytes": len(content),
                "tokenizer": "fixed UTF-8 bytes, vocabulary 256", "splits": records}
    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    (processed / f"splits_{digest(json.dumps(config, sort_keys=True).encode())[:12]}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return splits, manifest


def random_batch(data, batch_size, seq_len, generator, device):
    starts = torch.randint(len(data) - seq_len, (batch_size,), generator=generator)
    offsets = starts[:, None] + torch.arange(seq_len + 1)
    batch = data[offsets].to(device)
    return batch[:, :-1], batch[:, 1:]


def evaluation_batches(data, batch_size, seq_len, max_blocks, device):
    count = min((len(data) - 1) // seq_len, max_blocks)
    if count < 1:
        raise ValueError("Evaluation split too small")
    # No repeated prediction targets. Fixed blocks spread across each entire split.
    starts = torch.linspace(0, len(data) - seq_len - 1, count).long()
    for chunk in starts.split(batch_size):
        batch = data[chunk[:, None] + torch.arange(seq_len + 1)].to(device)
        yield batch[:, :-1], batch[:, 1:]
