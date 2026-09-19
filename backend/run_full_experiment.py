#!/usr/bin/env python3
"""
Požene celoten eksperiment: vloge x modeli x RAG x ponovitve.
Vsak surovi odgovor modela in oceno zagona sproti doda v
experiments/results.json, da se nič ne izgubi, če se kaj sesuje ali
naletimo na rate limit. test_data/ in knowledge_base/ pusti pri miru.
"""
import json
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))

from rag import RAGRetriever
import evaluate as ev

ROOT = Path(__file__).parent.parent
TEST_DIR = ROOT / "test_data"
OUT_DIR = ROOT / "experiments"
OUT_DIR.mkdir(exist_ok=True)
RESULTS_FILE = OUT_DIR / "results.json"

APPS = ["T1", "T2", "T3", "T4", "T5", "T6", "K1"]
# ključ iz MODEL_MAP -> ime, ki se izpiše v rezultatih/poročilih
MODEL_KEYS = [
    ("gpt-4.1", "gpt-4.1"),
    ("gpt-4o-mini", "gpt-4o-mini"),
    ("claude-3.5-sonnet", "claude-sonnet-4-6"),  # ključ je ostal star zaradi MODEL_MAP, v resnici teče claude-sonnet-4-6
]
RAG_OPTIONS = [True, False]
N_REPS = 3


def load_existing_results() -> list[dict]:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return []


def save_results(results: list[dict]) -> None:
    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def already_done(results: list[dict], app, model_label, use_rag, rep) -> bool:
    for r in results:
        if (r["app"] == app and r["model_label"] == model_label
                and r["rag"] == use_rag and r["rep"] == rep and r.get("error") is None):
            return True
    return False


def run_one(model_key: str, model_label: str, use_rag: bool, app_text: str,
            ground_truth: list[dict], boundary_cases: list[dict],
            retriever: RAGRetriever | None) -> dict:
    """V bistvu evaluate.run_check, le da vrne še surovi odgovor modela."""
    parts: list[str] = []
    if use_rag:
        if retriever is not None:
            chunks = retriever.retrieve(app_text, k=6)
            if chunks:
                provisions = "\n\n".join(
                    f"[{c['source']}{',' + c['article'] if c['article'] else ''}]\n{c['text']}"
                    for c in chunks
                )
                parts.append(f"RELEVANTNI PREDPISI:\n{provisions}")
    parts.append(f"VSEBINA VLOGE:\n{app_text}")
    parts.append(ev.JSON_INSTRUCTION)
    user_msg = "\n\n".join(parts)

    client = ev.get_model_client(model_key)
    raw = client.chat(system=ev.SYSTEM_PROMPT, user=user_msg, temperature=0)

    parsed = ev._parse_json(raw)
    predicted = [] if "error" in parsed else parsed.get("pomanjkljivosti", [])

    scored, boundary_matches = ev.filter_boundary(predicted, boundary_cases)
    tp_pairs, fp_list, fn_list = ev.match_deficiencies(scored, ground_truth)
    overall = ev.metrics(len(tp_pairs), len(fp_list), len(fn_list))

    per_cat = {}
    for cat in ev.CATEGORIES:
        tp_c = [(p, g, s) for p, g, s in tp_pairs if g.get("kategorija") == cat]
        fn_c = [x for x in fn_list if x.get("kategorija") == cat]
        fp_c = [x for x in fp_list if x.get("kategorija") == cat]
        per_cat[cat] = ev.metrics(len(tp_c), len(fp_c), len(fn_c))

    return {
        "raw_response": raw,
        "parse_error": parsed.get("error") if "error" in parsed else None,
        "predicted": predicted,
        "overall": overall,
        "per_cat": per_cat,
        "n_boundary_matches": len(boundary_matches),
        "n_tp": len(tp_pairs),
        "n_fp": len(fp_list),
        "n_fn": len(fn_list),
    }


def main() -> None:
    results = load_existing_results()
    print(f"Loaded {len(results)} existing results from {RESULTS_FILE.relative_to(ROOT).as_posix()}")

    retriever: RAGRetriever | None = None
    try:
        retriever = RAGRetriever()
        print("RAG retriever: OK")
    except Exception as exc:
        print(f"RAG retriever unavailable: {exc}")

    app_cache: dict[str, tuple] = {}
    for app in APPS:
        app_file = TEST_DIR / f"vloga_{app}.md"
        gt_file = TEST_DIR / f"referencna_resnica_{app}.json"
        app_cache[app] = ev.load_test_data(app_file, gt_file)

    total = len(APPS) * len(MODEL_KEYS) * len(RAG_OPTIONS) * N_REPS
    done_count = sum(1 for r in results if r.get("error") is None)
    print(f"Total planned runs: {total}  (already done: {done_count})")

    combo_i = 0
    for app in APPS:
        app_text, ground_truth, boundary_cases = app_cache[app]
        for model_key, model_label in MODEL_KEYS:
            for use_rag in RAG_OPTIONS:
                for rep in range(1, N_REPS + 1):
                    combo_i += 1
                    if already_done(results, app, model_label, use_rag, rep):
                        continue
                    tag = f"[{combo_i}/{total}] app={app} model={model_label} rag={'on' if use_rag else 'off'} rep={rep}"
                    print(tag, "... ", end="", flush=True)
                    t0 = time.time()
                    entry = {
                        "app": app,
                        "model_key": model_key,
                        "model_label": model_label,
                        "rag": use_rag,
                        "rep": rep,
                        "error": None,
                    }
                    try:
                        out = run_one(
                            model_key, model_label, use_rag,
                            app_text, ground_truth, boundary_cases, retriever,
                        )
                        entry.update(out)
                        m = out["overall"]
                        print(
                            f"TP={m['tp']} FP={m['fp']} FN={m['fn']} "
                            f"P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} "
                            f"({time.time()-t0:.1f}s)"
                        )
                    except Exception as exc:
                        entry["error"] = f"{type(exc).__name__}: {exc}"
                        entry["traceback"] = traceback.format_exc()
                        print(f"ERROR: {entry['error']}")

                    results.append(entry)
                    save_results(results)
                    time.sleep(0.4)

    print(f"\nDone. {len(results)} results saved to {RESULTS_FILE.relative_to(ROOT).as_posix()}")
    n_errors = sum(1 for r in results if r.get("error"))
    print(f"Errors: {n_errors}")


if __name__ == "__main__":
    main()
