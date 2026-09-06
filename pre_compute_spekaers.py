"""
pre_compute_spekaers.py -- target speaker embedding per speaker for the
speaker-consistency loss (AUX=1).

For each sid, averages ECAPA embeddings over that speaker's 16k slices
(<exp_dir>/sliced_audios_16k, written by preprocess -- already 16 kHz, so
no resampling is needed), L2-normalizes, and saves a [num_speakers, 192]
tensor to <exp_dir>/spk_targets.pt.

Usage:
    python pre_compute_spekaers.py logs/hindi_tts_base

Gate before training:
    python -c "import torch; t=torch.load('logs/hindi_tts_base/spk_targets.pt'); print(t.shape, t.norm(dim=1))"
    -> shape (num_speakers, 192) and EVERY norm exactly 1.0
"""

import os
import sys
from collections import defaultdict

import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

MAX_CLIPS_PER_SPEAKER = 200  # cap for speed; enough slices for a stable mean

exp_dir = os.path.abspath(sys.argv[1])
wav_dir = os.path.join(exp_dir, "sliced_audios_16k")
assert os.path.isdir(wav_dir), f"missing {wav_dir} -- run preprocess first"

device = "cuda" if torch.cuda.is_available() else "cpu"
enc = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}
)

# slices are named {sid}_{idx0}_{idx1}.wav by preprocess
by_sid = defaultdict(list)
for f in sorted(os.listdir(wav_dir)):
    if f.endswith(".wav"):
        by_sid[int(f.split("_")[0])].append(os.path.join(wav_dir, f))
assert by_sid, f"no wavs in {wav_dir}"
print("speakers found:", sorted(by_sid),
      "| files:", sum(len(v) for v in by_sid.values()))

targets = torch.zeros(max(by_sid) + 1, 192)
for sid, files in sorted(by_sid.items()):
    embs = []
    for fp in tqdm(files[:MAX_CLIPS_PER_SPEAKER], desc=f"sid {sid}"):
        try:
            d, sr = sf.read(fp, dtype="float32")
        except Exception as e:
            print(f"  skip (load fail): {fp} ({e})")
            continue
        if d.ndim == 2:
            d = d.mean(1)
        w = torch.from_numpy(d).unsqueeze(0)
        if sr != 16000:  # safety net only; this folder is already 16k
            w = torchaudio.functional.resample(w, sr, 16000)
        with torch.no_grad():
            embs.append(enc.encode_batch(w.to(device)).reshape(-1).cpu())
    assert embs, f"sid {sid}: every clip failed to load -- refusing to save a zero row"
    v = torch.stack(embs).mean(0)
    targets[sid] = v / v.norm()
    print(f"sid {sid}: averaged {len(embs)} clips, norm={float(targets[sid].norm()):.3f}")

out = os.path.join(exp_dir, "spk_targets.pt")
torch.save(targets, out)
print(f"saved {out}  shape={tuple(targets.shape)}  "
      f"norms={[round(float(x), 3) for x in targets.norm(dim=1)]}")
