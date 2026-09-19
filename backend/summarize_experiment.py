#!/usr/bin/env python3
"""
Iz experiments/results.json naredi tabele RV3, RV2 in K1.
Samo bere, test_data/ in knowledge_base/ pusti pri miru.
"""
import json
import statistics as st
from pathlib import Path

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers=(), tablefmt="grid"):  # type: ignore[misc]
        header_line = " | ".join(str(h) for h in headers)
        rows = [" | ".join(str(c) for c in row) for row in data]
        return header_line + "\n" + "\n".join(rows)

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "experiments" / "results.json"

ERROR_APPS = ["T1", "T2", "T3", "T4", "T5", "T6"]
MODEL_LABELS = ["gpt-4.1", "gpt-4o-mini", "claude-sonnet-4-6"]
CATEGORIES = ["A", "B", "C"]


def load_results() -> list[dict]:
    return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))


def f1_of(tp, fp, fn) -> float:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------------
# Združevanje: za dani model+RAG najprej povprečje čez 3 ponovitve, potem čez
# 6 vlog z napakami (T1-T6). Zraven še, kako enotne so ponovitve.
# ---------------------------------------------------------------------------

def per_app_rep_metrics(results, model_label, use_rag, app):
    rows = [r for r in results if r["model_label"] == model_label
            and r["rag"] == use_rag and r["app"] == app and r["error"] is None]
    rows.sort(key=lambda r: r["rep"])
    return rows


def aggregate(results, model_label, use_rag, apps):
    """Vrne dict s skupnimi P/R/F1, F1 po kategorijah in stabilnostjo.

    Vrstni red povprečenja: za vsako vlogo povprečje čez 3 ponovitve, nato
    povprečje čez vloge. Stabilnost = std. odklon F1 med ponovitvami znotraj
    vloge, povprečen čez vloge (0 = vsakič isto), plus delež vlog, kjer so
    vse 3 ponovitve dale točno iste TP/FP/FN.
    """
    per_app_overall_f1 = []
    per_app_p = []
    per_app_r = []
    per_app_f1 = []
    per_app_cat_f1 = {c: [] for c in CATEGORIES}
    per_app_std = []
    exact_agree_flags = []

    for app in apps:
        rows = per_app_rep_metrics(results, model_label, use_rag, app)
        if not rows:
            continue
        p_vals = [r["overall"]["precision"] for r in rows]
        r_vals = [r["overall"]["recall"] for r in rows]
        f1_vals = [r["overall"]["f1"] for r in rows]
        per_app_p.append(mean(p_vals))
        per_app_r.append(mean(r_vals))
        per_app_f1.append(mean(f1_vals))
        per_app_std.append(st.pstdev(f1_vals) if len(f1_vals) > 1 else 0.0)

        triples = {(r["n_tp"], r["n_fp"], r["n_fn"]) for r in rows}
        exact_agree_flags.append(1.0 if len(triples) == 1 else 0.0)

        for c in CATEGORIES:
            cat_f1_vals = [r["per_cat"][c]["f1"] for r in rows]
            per_app_cat_f1[c].append(mean(cat_f1_vals))

    return {
        "precision": mean(per_app_p),
        "recall": mean(per_app_r),
        "f1": mean(per_app_f1),
        "f1_A": mean(per_app_cat_f1["A"]),
        "f1_B": mean(per_app_cat_f1["B"]),
        "f1_C": mean(per_app_cat_f1["C"]),
        "f1_std_over_reps": mean(per_app_std),
        "exact_agreement_rate": mean(exact_agree_flags),
    }


def k1_fp(results, model_label, use_rag):
    rows = [r for r in results if r["model_label"] == model_label
            and r["rag"] == use_rag and r["app"] == "K1" and r["error"] is None]
    fps = [r["n_fp"] for r in rows]
    return mean(fps), fps


def main() -> None:
    results = load_results()
    print(f"Loaded {len(results)} result rows from {RESULTS_FILE.relative_to(ROOT).as_posix()}\n")

    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"WARNING: {len(errors)} runs had errors and are excluded from aggregation:")
        for e in errors:
            print(f"  {e['app']} {e['model_label']} rag={e['rag']} rep={e['rep']}: {e['error']}")
        print()

    # -----------------------------------------------------------------
    # TABELA RV3 (RAG on): vrstice = 3 modeli, povprečje čez ponovitve, nato čez T1-T6
    # -----------------------------------------------------------------
    print("=" * 100)
    print("TABLE RV3 — RAG ON, averaged over 3 repetitions then over T1-T6")
    print("=" * 100)
    rv3_rows = []
    for model_label in MODEL_LABELS:
        agg = aggregate(results, model_label, True, ERROR_APPS)
        rv3_rows.append([
            model_label,
            f"{agg['precision']:.3f}",
            f"{agg['recall']:.3f}",
            f"{agg['f1']:.3f}",
            f"{agg['f1_A']:.3f}",
            f"{agg['f1_B']:.3f}",
            f"{agg['f1_C']:.3f}",
            f"{agg['f1_std_over_reps']:.3f} (agree={agg['exact_agreement_rate']*100:.0f}%)",
        ])
    print(tabulate(
        rv3_rows,
        headers=["Model", "P", "R", "F1", "F1[A]", "F1[B]", "F1[C]",
                 "Stability (F1 std over reps / exact-agree rate)"],
        tablefmt="grid",
    ))
    print(
        "\nStability = mean, over T1-T6, of the population std-dev of F1 across the 3\n"
        "repetitions for that app (0.000 = repetitions always agreed on F1). exact-agree\n"
        "= share of apps where all 3 reps produced identical TP/FP/FN counts.\n"
    )

    # -----------------------------------------------------------------
    # TABELA RV2 (najboljši model): RAG on / off / razlika
    # -----------------------------------------------------------------
    best_model = max(MODEL_LABELS, key=lambda m: aggregate(results, m, True, ERROR_APPS)["f1"])
    print("=" * 100)
    print(f"TABLE RV2 — best model = {best_model} (highest RAG-on F1 in TABLE RV3), RAG on vs off")
    print("=" * 100)

    agg_on = aggregate(results, best_model, True, ERROR_APPS)
    agg_off = aggregate(results, best_model, False, ERROR_APPS)

    def diff(k):
        return agg_on[k] - agg_off[k]

    rv2_rows = [
        ["RAG on", f"{agg_on['precision']:.3f}", f"{agg_on['recall']:.3f}", f"{agg_on['f1']:.3f}",
         f"{agg_on['f1_A']:.3f}", f"{agg_on['f1_B']:.3f}", f"{agg_on['f1_C']:.3f}"],
        ["RAG off", f"{agg_off['precision']:.3f}", f"{agg_off['recall']:.3f}", f"{agg_off['f1']:.3f}",
         f"{agg_off['f1_A']:.3f}", f"{agg_off['f1_B']:.3f}", f"{agg_off['f1_C']:.3f}"],
        ["Difference (on - off)",
         f"{diff('precision'):+.3f}", f"{diff('recall'):+.3f}", f"{diff('f1'):+.3f}",
         f"{diff('f1_A'):+.3f}", f"{diff('f1_B'):+.3f}", f"{diff('f1_C'):+.3f}"],
    ]
    print(tabulate(rv2_rows, headers=["Config", "P", "R", "F1", "F1[A]", "F1[B]", "F1[C]"], tablefmt="grid"))
    print()

    # -----------------------------------------------------------------
    # K1 (kontrolna): povprečni FP za vsak model x RAG
    # -----------------------------------------------------------------
    print("=" * 100)
    print("K1 (control, 0 ground-truth deficiencies) — average false positives per model x RAG")
    print("=" * 100)
    k1_rows = []
    for model_label in MODEL_LABELS:
        for use_rag in (True, False):
            avg_fp, fps = k1_fp(results, model_label, use_rag)
            k1_rows.append([
                model_label,
                "on" if use_rag else "off",
                f"{avg_fp:.2f}",
                str(fps),
            ])
    print(tabulate(k1_rows, headers=["Model", "RAG", "Avg FP (3 reps)", "FP per rep"], tablefmt="grid"))


if __name__ == "__main__":
    main()
