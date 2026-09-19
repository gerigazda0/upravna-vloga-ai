# upravna-vloga-ai — preverjanje popolnosti vlog za gradbeno dovoljenje

Prototip, nastal v okviru diplomske naloge. Aplikacija sprejme dokumente vloge za
gradbeno dovoljenje (DGD), jih s pomočjo velikega jezikovnega modela (LLM) in
iskanja po predpisih (RAG) primerja z zahtevami ZUP, GZ-1, Pravilnika in Uredbe o
razvrščanju objektov ter vrne strukturirano poročilo o pomanjkljivostih:

| Kategorija | Pomen |
|:----------:|-------|
| **A** | Manjkajoča predpisana sestavina |
| **B** | Notranja neskladnost med deli vloge |
| **C** | Neskladnost z zahtevami predpisov |

Podprti so modeli GPT-4.1, GPT-4o-mini in Claude Sonnet; RAG je mogoče vklopiti ali izklopiti.

> Vse vloge v `test_data/` so **sintetične** — podatki so izmišljeni in namenjeni
> izključno testiranju. Repozitorij ne vsebuje realnih vlog ali osebnih podatkov.

---

## Arhitektura

```
React/Vite (frontend) ──HTTP──> FastAPI (backend) ──> LLM (OpenAI / Anthropic)
                                      │
                                      └──> Chroma (lokalna vektorska baza predpisov)
                                              ▲
                                  OpenAI text-embedding-3-small
```

- **Frontend** (`frontend/`) — React 18 + Vite; nalaganje dokumentov, izbira modela in RAG, prikaz poročila.
- **Backend** (`backend/`) — FastAPI strežnik (`main.py`), ekstrakcija besedila iz PDF/DOCX/TXT/MD (`text_extraction.py`), abstrakcija modelov (`models.py`), RAG iskanje (`rag.py`).
- **Vektorska baza** — Chroma (`chroma_db/`, lokalno, ni v repozitoriju). Besedila predpisov iz `knowledge_base/` se razrežejo na delce (~700 znakov) in vektorizirajo z modelom `text-embedding-3-small` (`ingest.py`).

### Struktura

```
├── backend/
│   ├── main.py                  # FastAPI strežnik
│   ├── models.py                # Modeli (OpenAI / Anthropic)
│   ├── rag.py                   # RAG iskanje (Chroma)
│   ├── text_extraction.py       # PDF/DOCX/TXT/MD -> besedilo
│   ├── ingest.py                # Gradnja vektorske baze iz knowledge_base/
│   ├── evaluate.py              # Evalvacija posamezne konfiguracije
│   ├── run_full_experiment.py   # Celoten eksperiment (126 zagonov)
│   ├── summarize_experiment.py  # Tabele RV2 / RV3 / K1 iz rezultatov
│   └── requirements.txt
├── frontend/                    # React/Vite aplikacija
├── knowledge_base/              # Besedila predpisov (vhod za RAG)
├── test_data/                   # Sintetične vloge T1–T6, kontrolna K1 + referenčna resnica
├── experiments/                 # Rezultati eksperimenta iz diplome
├── .env.example
└── LICENSE
```

---

## Zahteve

- Python 3.11+ (preizkušeno na 3.12)
- Node.js 18+
- API ključ **OpenAI** (GPT modeli **in** embeddings — potreben tudi, če uporabljate samo Claude)
- API ključ **Anthropic** (za Claude)

## Okoljske spremenljivke

Kopirajte `.env.example` v `.env` v korenu repozitorija in vpišite ključe:

```bash
cp .env.example .env
```

| Spremenljivka | Namen | Kje jo pridobiti |
|---------------|-------|------------------|
| `OPENAI_API_KEY` | GPT-4.1, GPT-4o-mini, embeddings za RAG | <https://platform.openai.com/api-keys> (zahteva račun z dobroimetjem) |
| `ANTHROPIC_API_KEY` | Claude Sonnet | <https://console.anthropic.com/settings/keys> |

Datoteka `.env` je v `.gitignore` — ključev nikoli ne commitajte.

---

## Namestitev

### Backend

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

### Vektorska baza (enkrat, oz. po spremembi predpisov)

```bash
cd backend
python ingest.py
```

## Zagon lokalno

V dveh terminalih:

```bash
# 1) backend — http://localhost:8000 (preverjanje: /health)
cd backend
uvicorn main:app --reload

# 2) frontend — http://localhost:5173
cd frontend
npm run dev
```

V brskalniku izberite model, nastavite RAG, naložite dokumente vloge in pritisnite **Preveri**.

---

## Eksperimenti (evalvacija, 126 zagonov)

Vse skripte poganjajte iz mape `backend/` z aktiviranim okoljem in zgrajeno vektorsko bazo.

### Celoten eksperiment iz diplome

```bash
python run_full_experiment.py
```

Mreža: 7 vlog (T1–T6 + kontrolna K1) × 3 modeli × RAG on/off × 3 ponovitve = **126 zagonov**.
Vsak surovi odgovor modela in ocena (TP/FP/FN, P, R, F1) se sproti zapisujeta v
`experiments/results.json`. Skripta je nadaljevalna: že uspešno opravljene zagone
preskoči, zato jo po prekinitvi (npr. rate limit) preprosto poženete znova.
Za povsem nov zagon najprej premaknite ali izbrišite `experiments/results.json`.

> Opomba: ključ `claude-3.5-sonnet` v `models.py` je ohranjen zaradi združljivosti,
> dejansko kliče model `claude-sonnet-4-6`.

### Povzetek rezultatov

```bash
python summarize_experiment.py
```

Iz `experiments/results.json` izpiše tabele RV3 (primerjava modelov z RAG),
RV2 (najboljši model, RAG on vs. off) in rezultate na kontrolni vlogi K1.
Rezultati iz diplome so shranjeni v `experiments/`.

### Posamezna konfiguracija

```bash
python evaluate.py --model gpt-4o-mini --rag on --test T1
python evaluate.py --model claude-3.5-sonnet --rag off --runs 3 --test T2
python evaluate.py --all-configs --runs 3 --no-verbose
```

Avtomatsko ujemanje napovedi z referenčno resnico temelji na prekrivanju ključnih
besed in je okvirno — razpis TP/FP/FN je smiselno preveriti ročno.

---

## Odpravljanje težav

| Simptom | Rešitev |
|---------|---------|
| `OPENAI_API_KEY not set` / `KeyError` | Preverite `.env` v korenu repozitorija |
| `RAG retriever ni dosegljiv`, `InvalidCollectionException` | Zaženite `python ingest.py` |
| CORS napaka v brskalniku | Preverite, da backend teče na portu 8000 |

## Licenca

[MIT](LICENSE) © 2026 Nejc Gerkšič
