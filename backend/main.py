"""
FastAPI backend za preverjevalnik DGD dokumentacije.
Zaženeš iz mape backend/:
    uvicorn main:app --reload
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

from models import MODEL_MAP, get_model_client
from rag import RAGRetriever
from text_extraction import extract_text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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


_rag_retriever: RAGRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_retriever
    try:
        _rag_retriever = RAGRetriever()
        log.info("RAG retriever uspešno inicializiran")
    except Exception as exc:
        log.warning("RAG retriever ni na voljo: %s", exc)
    yield


app = FastAPI(title="DGD Checker", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_available": _rag_retriever is not None,
        "models": list(MODEL_MAP.keys()),
    }


@app.post("/check")
async def check_application(
    files: List[UploadFile] = File(...),
    model: str = Form("gpt-4o-mini"),
    use_rag: bool = Form(True),
):
    # 1. Iz vseh naloženih dokumentov potegnemo besedilo in ga zlepimo skupaj
    parts = []
    for f in files:
        raw = await f.read()
        text = extract_text(raw, f.filename or "brez_imena")
        parts.append(f"=== {f.filename} ===\n{text}")
    application_content = "\n\n".join(parts)
    log.info("Extracted application text: %d chars from %d file(s)", len(application_content), len(files))

    # 2. Če je RAG vklopljen, poiščemo še relevantne kose predpisov
    user_parts: list[str] = []
    if use_rag:
        if _rag_retriever is not None:
            chunks = _rag_retriever.retrieve(application_content, k=6)
            if chunks:
                provisions = "\n\n".join(
                    f"[{c['source']}{',' + c['article'] if c['article'] else ''}]\n{c['text']}"
                    for c in chunks
                )
                user_parts.append(f"RELEVANTNI PREDPISI:\n{provisions}")
        else:
            log.warning("RAG zahtevan, toda retriever ni inicializiran – nadaljujem brez RAG")

    # 3. Sestavimo prompt
    user_parts.append(f"VSEBINA VLOGE:\n{application_content}")
    user_parts.append(JSON_INSTRUCTION)
    user_message = "\n\n".join(user_parts)
    log.info("Prompt to model %s: %d chars total (user message incl. RAG context)", model, len(user_message))

    # 4. Pokličemo izbrani model (temperatura 0)
    client = get_model_client(model)
    raw_response = client.chat(system=SYSTEM_PROMPT, user=user_message, temperature=0)

    # 5. Odgovor razčlenimo v JSON
    result = _parse_json(raw_response)
    return JSONResponse(content=result)
