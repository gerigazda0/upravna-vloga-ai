#!/usr/bin/env python3
"""
Skripta za evalvacijo DGD preverjevalnika (RV2 in RV3).

Par primerov (poženi iz mape backend/):
    python evaluate.py --model gpt-4o-mini --rag on --runs 3
    python evaluate.py --model claude-3.5-sonnet --rag off
    python evaluate.py --all-configs --runs 3
    python evaluate.py --all-configs --runs 1 --no-verbose
"""
import argparse
import json
import re
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent))

from models import MODEL_MAP, get_model_client
from rag import RAGRetriever

try:
    from tabulate import tabulate
except ImportError:
    print("Opozorilo: tabulate ni nameščen. Namestite z: pip install tabulate")
    def tabulate(data, headers=(), tablefmt="grid"):  # type: ignore[misc]
        header_line = " | ".join(str(h) for h in headers)
        rows = [" | ".join(str(c) for c in row) for row in data]
        return header_line + "\n" + "\n".join(rows)

TEST_DIR   = Path(__file__).parent.parent / "test_data"

SYSTEM_PROMPT = """\
Si strokovnjak za preverjanje popolnosti vlog za izdajo gradbenega dovoljenja. Predložene dokumentacije ne ocenjuješ na splošno, temveč izvedeš tri vrste preverjanja in vsako ugotovljeno pomanjkljivost uvrstiš v ustrezno kategorijo:

(A) Odsotnost predpisane sestavine. Preveri, ali so prisotne vse priloge in sestavine, ki jih za to vrsto vloge zahtevajo posredovani predpisi. Posebej preveri: ali je priloženo mnenje za vsak komunalni priključek, ki ga vloga navaja (npr. vodovod, kanalizacija, elektrika); ali je priloženo pooblastilo, kadar vlogo vlaga pooblaščenec; ali so priložena dokazila o pravici graditi, kadar so potrebna.

(B) Notranja neskladnost. Iz različnih delov vloge (obrazci, tehnično poročilo, situacije, prerezi, prikazi) izlušči iste podatke o objektu — dimenzije, višine, površine, naklone, barve, klasifikacijo, število enot ali parkirnih mest — in jih med seboj primerjaj. Vsako vrednost, ki je v enem delu vloge navedena drugače kot v drugem, prijavi kot neskladje ter navedi obe vrednosti in mesti, kjer se pojavita.

(C) Neskladnost z zunanjo zahtevo. Navedbe v vlogi primerjaj z zahtevami iz posredovanih predpisov. Pomanjkljivost te vrste prijavi le, kadar konkretna navedba v vlogi nasprotuje konkretni določbi iz posredovanih predpisov. Ne sklepaj o kršitvi meril, ki jih v posredovanih predpisih ni. Ne zamenjuj pojmov: »enostanovanjska stavba« (CC-SI 11100) je vrsta namembnosti in praviloma manj zahteven objekt, »enostavni objekt« pa je kategorija zahtevnosti; meril za enostavne objekte ne uporabljaj za stanovanjsko hišo, razen če vloga sama tako trdi.

Splošna pravila: ne izmišljaj si napak. Če podatka ni, tega ne štej za napako, razen če gre za predpisano sestavino, ki manjka. Vsako pomanjkljivost utemelji s konkretnim mestom v vlogi; pri kategorijah A in C navedi tudi konkretno določbo (člen in po možnosti odstavek ali točko) iz posredovanih predpisov, ne le imena datoteke. Odgovori izključno v predpisanem formatu JSON, brez dodatnega besedila.\
"""

JSON_INSTRUCTION = (
    'Vrni izključno JSON v obliki: '
    '{"pomanjkljivosti": [{'
    '"kategorija": "A/B/C", '
    '"opis": "kratek opis v slovenščini", '
    '"lokacija": "kje v vlogi", '
    '"pravna_podlaga": "npr. 55. člen ZUP / notranja neskladnost"'
    '}]}'
)

CATEGORIES = ["A", "B", "C"]
MATCH_THRESHOLD = 0.2   # najmanj toliko prekrivanja ključnih besed, da šteje kot ujemanje


def _parse_json(raw: str) -> dict:
    """Iz odgovora modela potegne JSON, tudi če je zavit v markdown ```.

    1. Odreže ```(json)...```, počisti presledke in proba json.loads.
    2. Če ne gre, vzame kos med prvim '{' in zadnjim '}'.
    3. Če tudi to ne gre, vrne {"error": ...}.
    """
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        clean = "\n".join(lines[1:end]).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    start, stop = raw.find("{"), raw.rfind("}")
    if start != -1 and stop > start:
        try:
            return json.loads(raw[start : stop + 1])
        except json.JSONDecodeError:
            pass
    return {"error": f"Model ni vrnil veljavnega JSON. Surovi odgovor: {raw[:300]}"}


# ---------------------------------------------------------------------------
# Osnovne pomožne funkcije
# ---------------------------------------------------------------------------

def load_test_data(app_file: Path, gt_file: Path) -> tuple[str, list[dict], list[dict]]:
    """Vrne (app_text, pomanjkljivosti, mejni_primeri).

    Če ključa mejni_primeri ni, dobiš [] (da starejši JSON-i še vedno delajo).
    """
    app_text = app_file.read_text(encoding="utf-8")
    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    return app_text, gt["pomanjkljivosti"], gt.get("mejni_primeri", [])


def run_check(
    app_text: str,
    model_name: str,
    use_rag: bool,
    retriever: RAGRetriever | None,
    print_raw: bool = False,
) -> list[dict]:
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
        else:
            print("  Opozorilo: RAG zahtevan, toda retriever ni dosegljiv.")

    parts.append(f"VSEBINA VLOGE:\n{app_text}")
    parts.append(JSON_INSTRUCTION)
    user_msg = "\n\n".join(parts)

    client = get_model_client(model_name)
    raw = client.chat(system=SYSTEM_PROMPT, user=user_msg, temperature=0)

    if print_raw:
        print("\n" + "=" * 80)
        print("RAW MODEL RESPONSE (nespremenjen)")
        print("=" * 80)
        print(raw)
        print("=" * 80 + "\n")

    parsed = _parse_json(raw)
    if "error" in parsed:
        print(f"  OPOZORILO: {parsed['error']}")
        return []
    return parsed.get("pomanjkljivosti", [])


_DIACRITICS = str.maketrans("šŠčČžŽ", "sScCzZ")

def normalize(text: str) -> str:
    """Vse v male črke, brez šumnikov in ločil, več presledkov stisne v enega."""
    text = text.lower().translate(_DIACRITICS)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_overlap(a: str, b: str) -> float:
    wa = {w for w in normalize(a).split() if len(w) > 3}
    wb = {w for w in normalize(b).split() if len(w) > 3}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def _score_matrix(rows: list[dict], cols: list[dict]) -> np.ndarray:
    """Zgradi matriko len(rows) × len(cols) s prekrivanjem ključnih besed."""
    mat = np.zeros((len(rows), len(cols)), dtype=float)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            mat[i, j] = keyword_overlap(r.get("opis", ""), c.get("opis", ""))
    return mat


def filter_boundary(
    predicted: list[dict], boundary_cases: list[dict]
) -> tuple[list[dict], list[tuple[dict, dict, float]]]:
    """Umakne na stran napovedi, ki se najbolje ujemajo z mejnim primerom.

    Uporabi linear_sum_assignment (čim večje skupno prekrivanje), tako da se
    vsaka napoved in vsak mejni primer porabita največ enkrat. Pari pod
    MATCH_THRESHOLD ne štejejo.

    Vrne (remaining_predicted, boundary_matches).
    boundary_matches = seznam (pred, boundary_gt, score)
    """
    if not predicted or not boundary_cases:
        return list(predicted), []

    mat = _score_matrix(predicted, boundary_cases)
    row_ind, col_ind = linear_sum_assignment(-mat)

    matched_pred: set[int] = set()
    boundary_matches: list[tuple[dict, dict, float]] = []
    for pi, bi in zip(row_ind, col_ind):
        score = mat[pi, bi]
        if score >= MATCH_THRESHOLD:
            matched_pred.add(pi)
            boundary_matches.append((predicted[pi], boundary_cases[bi], score))

    remaining = [p for i, p in enumerate(predicted) if i not in matched_pred]
    return remaining, boundary_matches


def match_deficiencies(
    predicted: list[dict], ground_truth: list[dict]
) -> tuple[list, list, list]:
    """Optimalno ujemanje ena-na-ena po vsebini (kategorija tu ni pomembna).

    linear_sum_assignment poišče razporeditev z največjim skupnim
    prekrivanjem, tako da se vsaka napoved in vsaka postavka GT porabita
    največ enkrat. Pari pod MATCH_THRESHOLD ne štejejo.

    Metrike po kategorijah se po ujemanju razvrstijo po kategoriji iz GT.

    Vrne (tp_pairs, fp_list, fn_list).
    tp_pairs = seznam (pred, gt, score)
    """
    if not predicted or not ground_truth:
        return [], list(predicted), list(ground_truth)

    mat = _score_matrix(predicted, ground_truth)
    row_ind, col_ind = linear_sum_assignment(-mat)

    matched_pred: set[int] = set()
    matched_gt:   set[int] = set()
    tp_pairs: list[tuple[dict, dict, float]] = []
    for pi, gi in zip(row_ind, col_ind):
        score = mat[pi, gi]
        if score >= MATCH_THRESHOLD:
            matched_pred.add(pi)
            matched_gt.add(gi)
            tp_pairs.append((predicted[pi], ground_truth[gi], score))

    fp = [predicted[i] for i in range(len(predicted)) if i not in matched_pred]
    fn = [ground_truth[i] for i in range(len(ground_truth)) if i not in matched_gt]
    return tp_pairs, fp, fn


def metrics(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Evalvacija enega zagona
# ---------------------------------------------------------------------------

def evaluate_once(
    model_name: str,
    use_rag: bool,
    app_text: str,
    ground_truth: list[dict],
    boundary_cases: list[dict],
    retriever: RAGRetriever | None,
    print_raw: bool = False,
) -> dict:
    predicted = run_check(app_text, model_name, use_rag, retriever, print_raw=print_raw)
    scored, boundary_matches = filter_boundary(predicted, boundary_cases)
    tp_pairs, fp_list, fn_list = match_deficiencies(scored, ground_truth)
    overall = metrics(len(tp_pairs), len(fp_list), len(fn_list))

    # Po kategorijah: TP/FN gresta po kategoriji iz GT, FP pa po kategoriji napovedi.
    per_cat: dict[str, dict] = {}
    for cat in CATEGORIES:
        tp_c = [(p, g, s) for p, g, s in tp_pairs if g.get("kategorija") == cat]
        fn_c = [x for x in fn_list if x.get("kategorija") == cat]
        fp_c = [x for x in fp_list if x.get("kategorija") == cat]
        per_cat[cat] = metrics(len(tp_c), len(fp_c), len(fn_c))

    return {
        "overall":          overall,
        "per_cat":          per_cat,
        "predicted":        predicted,
        "boundary_matches": boundary_matches,
        "tp_pairs":         tp_pairs,
        "fp_list":          fp_list,
        "fn_list":          fn_list,
    }


# ---------------------------------------------------------------------------
# Izpis za ljudi
# ---------------------------------------------------------------------------

def print_comparison(result: dict) -> None:
    print("\n" + "=" * 80)
    print("PRIMERJAVA: NAPOVED  vs.  REFERENČNA RESNICA")
    print("=" * 80)

    print("\n--- TRUE POSITIVES (pravilno zaznane pomanjkljivosti) ---")
    for i, (pred, gt, score) in enumerate(result["tp_pairs"]):
        cat_pred = pred.get("kategorija", "?")
        cat_gt   = gt.get("kategorija", "?")
        cat_note = f"{cat_pred}→{cat_gt}" if cat_pred != cat_gt else cat_pred
        print(f"\n  TP #{i+1}  (ujemanje = {score:.2f}  kategorija: {cat_note})")
        print(f"  NAPOVED  [{cat_pred}] {pred.get('opis')}")
        print(f"           Lokacija: {pred.get('lokacija')}  |  Podlaga: {pred.get('pravna_podlaga')}")
        print(f"  RESNICA  [{cat_gt}] {gt.get('opis')}")
        print(f"           Lokacija: {gt.get('lokacija')}  |  Podlaga: {gt.get('pravna_podlaga')}")

    print("\n--- FALSE POSITIVES (napačno zaznane – ni jih v referenčni resnici) ---")
    for i, fp in enumerate(result["fp_list"]):
        print(f"\n  FP #{i+1}  [{fp.get('kategorija')}] {fp.get('opis')}")
        print(f"           Lokacija: {fp.get('lokacija')}  |  Podlaga: {fp.get('pravna_podlaga')}")

    print("\n--- FALSE NEGATIVES (spregledane pomanjkljivosti iz referenčne resnice) ---")
    for i, fn in enumerate(result["fn_list"]):
        print(f"\n  FN #{i+1}  [{fn.get('kategorija')}] {fn.get('opis')}")
        print(f"           Lokacija: {fn.get('lokacija')}  |  Podlaga: {fn.get('pravna_podlaga')}")

    print("\n--- Mejni primeri (nešteti) ---")
    boundary_matches = result.get("boundary_matches", [])
    if not boundary_matches:
        print("\n  (ni mejnih zadetkov)")
    for i, (pred, bc, score) in enumerate(boundary_matches):
        print(f"\n  MEJNI #{i+1}  (ujemanje = {score:.2f})")
        print(f"  NAPOVED  [{pred.get('kategorija')}] {pred.get('opis')}")
        print(f"           Lokacija: {pred.get('lokacija')}  |  Podlaga: {pred.get('pravna_podlaga')}")
        print(f"  MEJNI GT [{bc.get('kategorija')}] {bc.get('opis')}")
        print(f"           Lokacija: {bc.get('lokacija')}  |  Podlaga: {bc.get('pravna_podlaga')}")

    print("\n" + "=" * 80)
    print("OPOMBA: Avtomatsko ujemanje temelji na prekrivanju ključnih besed.")
    print("Preverite TP/FP/FN ročno za dokončno klasifikacijo.")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Več zagonov ene konfiguracije
# ---------------------------------------------------------------------------

def run_config(
    model_name: str,
    use_rag: bool,
    n_runs: int,
    app_text: str,
    ground_truth: list[dict],
    boundary_cases: list[dict],
    retriever: RAGRetriever | None,
    verbose: bool = True,
) -> dict | None:
    label = f"{model_name} / RAG={'VKLOPLJEN' if use_rag else 'IZKLOPLJEN'}"
    print(f"\n{'='*60}")
    print(f"  Konfiguracija: {label}")
    print(f"  Ponovitve: {n_runs}")
    print("=" * 60)

    all_metrics: list[dict] = []
    last_result: dict | None = None

    for run_i in range(n_runs):
        print(f"  Ponovitev {run_i + 1}/{n_runs} … ", end="", flush=True)
        try:
            res = evaluate_once(
                model_name, use_rag, app_text,
                ground_truth, boundary_cases, retriever,
                print_raw=(verbose and run_i == 0),
            )
            all_metrics.append(res["overall"])
            last_result = res
            m = res["overall"]
            bm = len(res["boundary_matches"])
            print(
                f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  MEJNI={bm}  "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
            )
            if run_i < n_runs - 1:
                time.sleep(0.5)
        except Exception as exc:
            print(f"NAPAKA: {exc}")

    if not all_metrics:
        return None

    n = len(all_metrics)
    avg = {
        k: sum(m[k] for m in all_metrics) / n
        for k in ("precision", "recall", "f1", "tp", "fp", "fn")
    }

    print(f"\n  Povprečje ({n} ponovitev):")
    print(f"    P={avg['precision']:.3f}  R={avg['recall']:.3f}  F1={avg['f1']:.3f}")
    print(f"    TP={avg['tp']:.1f}  FP={avg['fp']:.1f}  FN={avg['fn']:.1f}")

    if last_result and verbose:
        print("\n  Po kategorijah (zadnja ponovitev):")
        for cat in CATEGORIES:
            cm = last_result["per_cat"][cat]
            print(
                f"    [{cat}]  P={cm['precision']:.3f}  R={cm['recall']:.3f}  "
                f"F1={cm['f1']:.3f}  (TP={cm['tp']} FP={cm['fp']} FN={cm['fn']})"
            )
        print_comparison(last_result)

    return {
        "label":    label,
        "model":    model_name,
        "use_rag":  use_rag,
        "n":        n,
        "avg":      avg,
        "last_cat": last_result["per_cat"] if last_result else {},
    }


# ---------------------------------------------------------------------------
# Vstopna točka
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evalvacija DGD preverjevalnika")
    parser.add_argument(
        "--model", choices=list(MODEL_MAP), default="gpt-4o-mini",
        help="Model (privzeto: gpt-4o-mini)",
    )
    parser.add_argument(
        "--rag", choices=["on", "off"], default="on",
        help="RAG vklopljen ali izklopljen (privzeto: on)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Število ponovitev (privzeto: 1)",
    )
    parser.add_argument(
        "--all-configs", action="store_true",
        help="Poženite vse kombinacije (3 modeli × RAG on/off)",
    )
    parser.add_argument(
        "--no-verbose", action="store_true",
        help="Izpusti podrobno primerjavo",
    )
    parser.add_argument(
        "--test", default="T1",
        help="Testna vloga (npr. T1 ali T2; privzeto: T1)",
    )
    args = parser.parse_args()

    app_file = TEST_DIR / f"vloga_{args.test}.md"
    gt_file  = TEST_DIR / f"referencna_resnica_{args.test}.json"

    print("Nalagam testne podatke …")
    app_text, ground_truth, boundary_cases = load_test_data(app_file, gt_file)
    print(f"  Vloga: {len(app_text):,} znakov")
    print(
        f"  Referenčna resnica: {len(ground_truth)} pomanjkljivosti "
        f"({sum(1 for g in ground_truth if g['kategorija']=='A')} A, "
        f"{sum(1 for g in ground_truth if g['kategorija']=='B')} B, "
        f"{sum(1 for g in ground_truth if g['kategorija']=='C')} C)"
    )
    if boundary_cases:
        print(
            f"  Mejni primeri: {len(boundary_cases)} "
            f"({sum(1 for g in boundary_cases if g.get('kategorija')=='A')} A, "
            f"{sum(1 for g in boundary_cases if g.get('kategorija')=='B')} B, "
            f"{sum(1 for g in boundary_cases if g.get('kategorija')=='C')} C)"
        )

    retriever: RAGRetriever | None = None
    try:
        retriever = RAGRetriever()
        print("  RAG retriever: OK")
    except Exception as exc:
        print(f"  RAG retriever ni dosegljiv: {exc}")

    verbose = not args.no_verbose

    if args.all_configs:
        configs = [
            (model, rag_on)
            for model in MODEL_MAP
            for rag_on in (True, False)
        ]
        results = []
        for model_name, use_rag in configs:
            r = run_config(
                model_name, use_rag, args.runs,
                app_text, ground_truth, boundary_cases, retriever,
                verbose=verbose,
            )
            if r:
                results.append(r)

        if results:
            print("\n" + "=" * 80)
            print("PRIMERJALNA TABELA VSEH KONFIGURACIJ")
            print("=" * 80)
            table = []
            for r in results:
                a = r["avg"]
                table.append([
                    r["model"],
                    "DA" if r["use_rag"] else "NE",
                    r["n"],
                    f"{a['precision']:.3f}",
                    f"{a['recall']:.3f}",
                    f"{a['f1']:.3f}",
                    f"{a['tp']:.1f}",
                    f"{a['fp']:.1f}",
                    f"{a['fn']:.1f}",
                ])
            headers = ["Model", "RAG", "N", "Precision", "Recall", "F1", "TP", "FP", "FN"]
            print(tabulate(table, headers=headers, tablefmt="grid"))
    else:
        run_config(
            args.model, args.rag == "on", args.runs,
            app_text, ground_truth, boundary_cases, retriever,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
