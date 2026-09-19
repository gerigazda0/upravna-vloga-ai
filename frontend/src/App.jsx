import { useEffect, useRef, useState } from 'react'

const MODELS = [
  { value: 'gpt-4.1',          label: 'GPT-4.1' },
  { value: 'gpt-4o-mini',      label: 'GPT-4o-mini' },
  { value: 'claude-3.5-sonnet', label: 'Claude 4.6 Sonnet' },
]

const CATEGORY_META = {
  A: { label: 'Manjkajoča sestavina',      color: 'var(--cat-a)', bg: 'var(--cat-a-bg)' },
  B: { label: 'Notranja neskladnost',       color: 'var(--cat-b)', bg: 'var(--cat-b-bg)' },
  C: { label: 'Neskladnost s predpisom',    color: 'var(--cat-c)', bg: 'var(--cat-c-bg)' },
}

function getInitialTheme() {
  if (typeof document !== 'undefined') {
    const attr = document.documentElement.getAttribute('data-theme')
    if (attr === 'light' || attr === 'dark') return attr
  }
  return 'light'
}

function Badge({ cat }) {
  const meta = CATEGORY_META[cat] ?? { color: '#555', bg: '#eee', label: cat }
  return (
    <span
      className="cat-badge"
      style={{ background: meta.color, color: '#fff' }}
      title={meta.label}
    >
      {cat}
    </span>
  )
}

function DeficiencyCard({ item, index }) {
  const meta = CATEGORY_META[item.kategorija] ?? { color: '#888', bg: '#f5f5f5', label: '' }
  return (
    <div className="card" style={{ borderLeftColor: meta.color, background: meta.bg }}>
      <div className="card-header">
        <Badge cat={item.kategorija} />
        <span className="card-cat-label">{meta.label}</span>
        <span className="card-index">#{index + 1}</span>
      </div>
      <p className="card-desc">{item.opis}</p>
      <dl className="card-meta">
        <dt>Lokacija</dt>
        <dd>{item.lokacija}</dd>
        <dt>Pravna podlaga</dt>
        <dd>{item.pravna_podlaga}</dd>
      </dl>
    </div>
  )
}

function Summary({ items, hiddenCats, onToggle, onReset }) {
  const counts = { A: 0, B: 0, C: 0 }
  items.forEach(i => { if (i.kategorija in counts) counts[i.kategorija]++ })
  return (
    <div className="summary-row">
      {Object.entries(counts).map(([cat, n]) => (
        <button
          key={cat}
          type="button"
          className={`summary-chip ${hiddenCats.has(cat) ? 'inactive' : ''}`}
          style={{ borderColor: CATEGORY_META[cat].color, color: CATEGORY_META[cat].color }}
          onClick={() => onToggle(cat)}
          title={`Klikni za ${hiddenCats.has(cat) ? 'prikaz' : 'skritje'} kategorije ${cat}`}
        >
          <Badge cat={cat} />&nbsp;{n}
        </button>
      ))}
      {hiddenCats.size > 0 && (
        <button type="button" className="summary-reset" onClick={onReset}>Prikaži vse</button>
      )}
      <div className="summary-total">Skupaj: <strong>{items.length}</strong></div>
    </div>
  )
}

export default function App() {
  const [model, setModel]       = useState('gpt-4o-mini')
  const [useRag, setUseRag]     = useState(true)
  const [files, setFiles]       = useState([])
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [result, setResult]     = useState(null)
  const [error, setError]       = useState(null)
  const [theme, setTheme]       = useState(getInitialTheme)
  const [hiddenCats, setHiddenCats] = useState(new Set())
  const inputRef = useRef(null)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('dgd-theme', theme)
  }, [theme])

  function toggleCat(cat) {
    setHiddenCats(prev => {
      const next = new Set(prev)
      next.has(cat) ? next.delete(cat) : next.add(cat)
      return next
    })
  }

  function handleFiles(fileList) {
    setFiles(Array.from(fileList))
    setResult(null)
    setError(null)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (files.length === 0) {
      setError('Izberite vsaj eno datoteko vloge.')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    setHiddenCats(new Set())

    const fd = new FormData()
    files.forEach(f => fd.append('files', f))
    fd.append('model', model)
    fd.append('use_rag', useRag)

    try {
      const resp = await fetch('/check', { method: 'POST', body: fd })
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}))
        throw new Error(err.detail || `Napaka strežnika: ${resp.status}`)
      }
      setResult(await resp.json())
    } catch (err) {
      setError(err.message || 'Prišlo je do neznane napake.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="site-header">
        <div className="header-inner">
          <div className="header-left">
            <div className="header-logo">DGD</div>
            <div>
              <h1>Preverjevalnik popolnosti vloge</h1>
              <p className="header-sub">Preverjanje DGD dokumentacije za gradbeno dovoljenje z LLM in RAG</p>
            </div>
          </div>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))}
            aria-label="Preklopi svetlo/temno temo"
            title="Preklopi svetlo/temno temo"
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
      </header>

      <main className="main">
        {/* ── Nastavitve ── */}
        <form className="panel form-panel" onSubmit={handleSubmit}>
          <h2 className="panel-title">Nastavitve preverjanja</h2>

          <div className="field-row">
            <label htmlFor="model-sel" className="field-label">Model</label>
            <select
              id="model-sel"
              className="select"
              value={model}
              onChange={e => setModel(e.target.value)}
              disabled={loading}
            >
              {MODELS.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </div>

          <div className="field-row toggle-row">
            <label className="toggle-label">
              <div className={`toggle ${useRag ? 'on' : ''}`} onClick={() => !loading && setUseRag(v => !v)}>
                <div className="toggle-thumb" />
              </div>
              <span>
                <strong>RAG</strong> – iskanje po predpisih{' '}
                <span className={`rag-status ${useRag ? 'rag-on' : 'rag-off'}`}>
                  {useRag ? 'VKLOPLJENO' : 'IZKLOPLJENO'}
                </span>
              </span>
            </label>
          </div>

          <div className="field-row">
            <label className="field-label">Dokumenti vloge</label>
            <div
              className={`drop-zone ${dragging ? 'dragging' : ''} ${files.length ? 'has-files' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.md"
                style={{ display: 'none' }}
                onChange={e => handleFiles(e.target.files)}
              />
              {files.length === 0 ? (
                <>
                  <div className="drop-icon">📂</div>
                  <p>Kliknite ali povlecite datoteke sem</p>
                  <p className="drop-hint">Podprti formati: PDF, DOCX, TXT, MD</p>
                </>
              ) : (
                <>
                  <div className="drop-icon">📄</div>
                  <ul className="file-list">
                    {files.map((f, i) => (
                      <li key={i}>{f.name} <span className="file-size">({(f.size / 1024).toFixed(1)} KB)</span></li>
                    ))}
                  </ul>
                  <p className="drop-hint">Kliknite za zamenjavo</p>
                </>
              )}
            </div>
          </div>

          <button className="btn-primary" type="submit" disabled={loading}>
            {loading
              ? <><span className="spinner-inline" /> Preverjam …</>
              : '✔ Preveri dokumentacijo'}
          </button>
        </form>

        {/* ── Rezultati ── */}
        <section className="results-area">
          {error && (
            <div className="panel error-panel">
              <strong>Napaka:</strong> {error}
            </div>
          )}

          {loading && (
            <div className="panel loading-panel">
              <div className="spinner-lg" />
              <p>Model pregleduje dokumentacijo …<br /><small>To lahko traja do 30 sekund.</small></p>
            </div>
          )}

          {result && !loading && (
            <div className="panel result-panel">
              <h2 className="panel-title">Rezultati preverjanja</h2>

              {result.pomanjkljivosti?.length > 0 ? (
                <>
                  <Summary
                    items={result.pomanjkljivosti}
                    hiddenCats={hiddenCats}
                    onToggle={toggleCat}
                    onReset={() => setHiddenCats(new Set())}
                  />
                  <div className="legend">
                    {Object.entries(CATEGORY_META).map(([cat, m]) => (
                      <span key={cat} className="legend-item">
                        <Badge cat={cat} /> {m.label}
                      </span>
                    ))}
                  </div>
                  {(() => {
                    const visible = result.pomanjkljivosti
                      .map((item, i) => ({ item, i }))
                      .filter(({ item }) => !hiddenCats.has(item.kategorija))
                    return visible.length > 0 ? (
                      <div className="card-list">
                        {visible.map(({ item, i }) => (
                          <DeficiencyCard key={i} item={item} index={i} />
                        ))}
                      </div>
                    ) : (
                      <div className="no-issues">
                        <div className="no-issues-icon">🔍</div>
                        <p>Ni pomanjkljivosti za izbrane filtre.</p>
                      </div>
                    )
                  })()}
                </>
              ) : (
                <div className="no-issues">
                  <div className="no-issues-icon">✅</div>
                  <p>V pregledani dokumentaciji niso bile zaznane pomanjkljivosti.</p>
                </div>
              )}

              {result.raw_response && (
                <details className="raw-details">
                  <summary>Surovi odgovor modela (debug)</summary>
                  <pre>{result.raw_response}</pre>
                </details>
              )}
            </div>
          )}

          {!result && !loading && !error && (
            <div className="panel empty-panel">
              <div className="empty-icon">🏗️</div>
              <p>Izberite model, nastavite RAG, naložite dokumente vloge in pritisnite <strong>Preveri</strong>.</p>
            </div>
          )}
        </section>
      </main>

      <footer className="site-footer">
        Prototip za diplomsko nalogo — Preverjevalnik DGD s pomočjo LLM in RAG
      </footer>
    </div>
  )
}
