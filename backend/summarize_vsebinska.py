#!/usr/bin/env python3
"""
Končne mere iz diplome na podlagi vsebinske razvrstitve javljenih pomanjkljivosti.

Samodejno ujemanje v evaluate.py (prekrivanje ključnih besed) je služilo le kot
prvo sito. Končna razvrstitev vsake različne javljene pomanjkljivosti je v
experiments/vsebinska_razvrstitev.json; ta skripta iz nje in iz surovih odgovorov
v experiments/results.json izračuna RP/LP/LN (prirejanje ena na ena), mere
P, R, F1 (in F2) ter okvirne 95-odstotne intervale zaupanja.

Zaženeš (iz mape backend/):
    python summarize_vsebinska.py
Ne potrebuje API ključev.
"""
import json
import math
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = json.loads((ROOT / "experiments" / "results.json").read_text(encoding="utf-8"))
LABELS = json.loads((ROOT / "experiments" / "vsebinska_razvrstitev.json").read_text(encoding="utf-8"))["postavke"]
APPS = ["T1", "T2", "T3", "T4", "T5", "T6"]
MODELS = [("claude-sonnet-4-6", "Claude Sonnet 4.6"), ("gpt-4.1", "GPT-4.1"), ("gpt-4o-mini", "GPT-4o-mini")]

GT = {}
for a in APPS:
    g = json.loads((ROOT / "test_data" / f"referencna_resnica_{a}.json").read_text(encoding="utf-8"))
    GT[a] = {x["id"]: x["kategorija"] for x in g["pomanjkljivosti"]}
LAB = {(x["vloga"], x["opis"]): x["razvrstitev"] for x in LABELS}


def match(cands):
    """Največje prirejanje ena na ena (javljena postavka -> ID napake)."""
    owner = {}

    def try_assign(i, seen):
        for g in cands[i]:
            if g in seen:
                continue
            seen.add(g)
            if g not in owner or try_assign(owner[g], seen):
                owner[g] = i
                return True
        return False

    for i in range(len(cands)):
        try_assign(i, set())
    return owner  # id -> indeks postavke


def score_run(r):
    gt = GT[r["app"]]
    preds = []
    for p in r["predicted"] or []:
        lab = LAB[(r["app"], p["opis"])]
        if lab == "mejni":
            continue
        preds.append((p, [] if lab == "LP" else lab))
    owner = match([c for _, c in preds])
    matched = set(owner.values())
    fp = [preds[i][0] for i in range(len(preds)) if i not in matched]
    fn = [g for g in gt if g not in owner]
    pc = {}
    for c in "ABC":
        pc[c] = (sum(1 for g in owner if gt[g] == c), sum(1 for p in fp if p.get("kategorija") == c),
                 sum(1 for g in fn if gt[g] == c))
    return len(owner), len(fp), len(fn), pc


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


RUNS = [dict(r, s=score_run(r)) for r in RESULTS if r["app"] in APPS]


def rows(model, rag, app=None):
    return [r for r in RUNS if r["model_label"] == model and r["rag"] == rag and (app is None or r["app"] == app)]


def aggregate(model, rag):
    P, R, F, FC, SD, AG = [], [], [], {c: [] for c in "ABC"}, [], []
    for a in APPS:
        rr = rows(model, rag, a)
        v = [prf(*x["s"][:3]) for x in rr]
        P.append(st.mean(t[0] for t in v)); R.append(st.mean(t[1] for t in v)); F.append(st.mean(t[2] for t in v))
        SD.append(st.pstdev([t[2] for t in v])); AG.append(len({x["s"][:3] for x in rr}) == 1)
        for c in "ABC":
            FC[c].append(st.mean(prf(*x["s"][3][c])[2] for x in rr))
    out = dict(P=st.mean(P), R=st.mean(R), F1=st.mean(F), std=st.mean(SD), agree=sum(AG) / len(AG),
               appF1=dict(zip(APPS, F)), **{f"F1{c}": st.mean(FC[c]) for c in "ABC"})
    out["F2"] = 5 * out["P"] * out["R"] / (4 * out["P"] + out["R"])
    return out


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def boot(vals, B=10000, seed=1):
    rnd = random.Random(seed)
    xs = sorted(st.mean(rnd.choice(vals) for _ in vals) for _ in range(B))
    return xs[int(0.025 * B)], xs[int(0.975 * B)]


f = lambda x: f"{x:.2f}".replace(".", ",")

print("TABELA 5 — primerjava modelov (RAG vklopljen)")
print("Model              P     R     F1    F1[A] F1[B] F1[C] F2    stabilnost")
AGG = {(m, rag): aggregate(m, rag) for m, _ in MODELS for rag in (True, False)}
for m, lab in MODELS:
    a = AGG[(m, True)]
    print(f"{lab:18} {f(a['P'])}  {f(a['R'])}  {f(a['F1'])}  {f(a['F1A'])}  {f(a['F1B'])}  {f(a['F1C'])}  {f(a['F2'])}  std {a['std']:.3f}; {a['agree']*100:.0f} %".replace('std 0.', 'std 0,'))

print("\nTABELA 6 — vpliv RAG (z / brez / razlika)")
for m, lab in MODELS:
    on, off = AGG[(m, True)], AGG[(m, False)]
    print(lab)
    for k in ["P", "R", "F1", "F1A", "F1B", "F1C"]:
        print(f"  {k:4} {f(on[k])}  {f(off[k])}  {on[k]-off[k]:+.2f}".replace(".", ","))

print("\nTABELA 7 — Claude Sonnet 4.6 (RAG) po vlogah: RP LP LN P R F1")
for a in APPS:
    rr = rows("claude-sonnet-4-6", True, a)
    v = [prf(*x["s"][:3]) for x in rr]
    tp, fp, fn = (st.mean(x["s"][i] for x in rr) for i in range(3))
    print(a, f(tp), f(fp), f(fn), f(st.mean(t[0] for t in v)), f(st.mean(t[1] for t in v)), f(st.mean(t[2] for t in v)))

print("\nK1 — lažna opozorila na kontrolni vlogi")
for m, lab in MODELS:
    for rag in (True, False):
        fps = [len(r["predicted"] or []) for r in RESULTS if r["app"] == "K1" and r["model_label"] == m and r["rag"] == rag]
        print(f"  {lab:18} RAG={'on ' if rag else 'off'} {fps}")

print("\nOkvirni 95-% intervali zaupanja (enota vzorčenja: vloga)")
for m, lab in MODELS:
    for rag in (True, False):
        rr = rows(m, rag)
        TP, FP, FN = (sum(x["s"][i] for x in rr) / 3 for i in range(3))
        lo, hi = boot(list(AGG[(m, rag)]["appF1"].values()))
        rl, rh = wilson(round(TP), round(TP + FN))
        print(f"  {lab:18} RAG={'on ' if rag else 'off'} F1 {f(AGG[(m, rag)]['F1'])} [{f(lo)}; {f(hi)}]  R (Wilson) [{f(rl)}; {f(rh)}]")
def pdiff(a, b):
    d = [AGG[a]["appF1"][x] - AGG[b]["appF1"][x] for x in APPS]
    lo, hi = boot(d)
    return f"{st.mean(d):+.2f} [{lo:+.2f}; {hi:+.2f}]".replace(".", ",")
print("  Claude − GPT-4.1 (RAG):     ", pdiff(("claude-sonnet-4-6", True), ("gpt-4.1", True)))
print("  GPT-4.1 − GPT-4o-mini (RAG):", pdiff(("gpt-4.1", True), ("gpt-4o-mini", True)))
print("  Claude − GPT-4o-mini (RAG): ", pdiff(("claude-sonnet-4-6", True), ("gpt-4o-mini", True)))
print("  Claude: z RAG − brez RAG:   ", pdiff(("claude-sonnet-4-6", True), ("claude-sonnet-4-6", False)))
