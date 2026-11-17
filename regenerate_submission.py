"""Rebuild the False Five submission byte for byte from the saved decoder inputs.

The training pipeline (notebooks/false_five_possession.ipynb) ends with a
deterministic step: a duration-constrained three-state Viterbi decode of the
ensemble probabilities. This script runs exactly that step from the three files
the notebook saves, so the submitted CSV can be reproduced on any machine in
seconds, without a GPU and without retraining.

Usage:
    python regenerate_submission.py
    python regenerate_submission.py --artifacts artifacts --out False_Five.csv
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True)
def viterbi_dur(lp, cost, kmin):
    """Viterbi over H, A, D where state k must last at least kmin[k] frames.

    Each state k is expanded into a chain of kmin[k] sub-states, so the minimum
    duration is enforced exactly (a semi-Markov minimum-duration constraint).
    """
    T = lp.shape[0]
    off = np.zeros(4, np.int64)
    for k in range(3):
        off[k + 1] = off[k] + kmin[k]
    S = off[3]
    lab = np.zeros(S, np.int64)
    for k in range(3):
        for i in range(kmin[k]):
            lab[off[k] + i] = k
    NEG = -1e18
    sc = np.full(S, NEG)
    for k in range(3):
        sc[off[k]] = lp[0, k]
    back = np.zeros((T, S), np.int32)
    new = np.empty(S)
    for t in range(1, T):
        for j in range(3):
            best = NEG
            bi = -1
            for i in range(3):
                if i == j:
                    continue
                v = sc[off[i + 1] - 1] - cost[i, j]
                if v > best:
                    best = v
                    bi = off[i + 1] - 1
            if kmin[j] == 1:
                if sc[off[j]] >= best:
                    best = sc[off[j]]
                    bi = off[j]
            new[off[j]] = best + lp[t, j]
            back[t, off[j]] = bi
            for s in range(1, kmin[j]):
                st = off[j] + s
                if s == kmin[j] - 1 and sc[st] > sc[st - 1]:
                    new[st] = sc[st] + lp[t, j]
                    back[t, st] = st
                else:
                    new[st] = sc[st - 1] + lp[t, j]
                    back[t, st] = st - 1
        for s in range(S):
            sc[s] = new[s]
    bs = 0
    for s in range(S):
        if sc[s] > sc[bs]:
            bs = s
    path = np.zeros(T, np.int8)
    s = bs
    for t in range(T - 1, -1, -1):
        path[t] = lab[s]
        if t > 0:
            s = back[t, s]
    return path


def decode(artifacts: Path) -> np.ndarray:
    cfg = json.load(open(artifacts / "decoder.json"))
    prob = np.load(artifacts / "final_probabilities.npy").astype(np.float64)
    lp = np.log(np.clip(prob, 1e-4, 1.0))
    lp[:, 2] += np.load(artifacts / "decoder_evidence.npy").astype(np.float64)
    ct, cd = cfg["c_team"], cfg["c_dead"]
    cost = np.array([[0, ct, cd], [ct, 0, cd], [cd, cd, 0]], np.float64)
    kmin = np.array([1, 1, cfg["min_dead"]], np.int64)
    out = np.full(cfg["n_frames"], 3, np.int8)
    for s, e in cfg["halves"]:
        out[s:e] = viterbi_dur(np.ascontiguousarray(lp[s:e]), cost, kmin)
    return np.array(list("HADX"))[out]


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", type=Path, default=root / "artifacts")
    ap.add_argument("--out", type=Path, default=root / "False_Five_regenerated.csv")
    ap.add_argument("--reference", type=Path, default=root / "submission" / "False_Five.csv",
                    help="submitted file to compare against (skipped if missing)")
    args = ap.parse_args()

    labels = decode(args.artifacts)
    with open(args.out, "w", newline="\n") as f:
        f.write("frame,label\n")
        f.write("".join(f"{i},{lab}\n" for i, lab in enumerate(labels)))

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"wrote {args.out}  ({len(labels)} rows)")
    print(f"sha256 {digest}")
    if args.reference.exists():
        same = args.out.read_bytes() == args.reference.read_bytes()
        print(f"byte-identical to {args.reference.name}: {same}")
        if not same:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
