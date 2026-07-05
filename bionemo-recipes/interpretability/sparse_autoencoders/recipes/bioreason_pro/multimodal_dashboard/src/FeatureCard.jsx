import React, { useState, useEffect, useRef, forwardRef } from 'react'
import ProteinSequence, { computeAlignInfo } from './ProteinSequence'
import MolstarThumbnail from './MolstarThumbnail'
import ProteinDetailModal from './ProteinDetailModal'
import { getAccession, uniprotUrl } from './utils'

const styles = {
  card: {
    background: 'var(--bg-card)',
    borderRadius: '8px',
    border: '1px solid var(--border)',
    flexShrink: 0,
  },
  cardHighlighted: {
    background: 'var(--bg-card)',
    borderRadius: '8px',
    border: '2px solid var(--highlight-border)',
    flexShrink: 0,
    boxShadow: '0 2px 8px var(--highlight-shadow)',
  },
  header: {
    padding: '12px 14px',
    borderBottom: '1px solid var(--border-light)',
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '10px',
  },
  headerLeft: {
    flex: 1,
    minWidth: 0,
  },
  featureId: {
    fontSize: '11px',
    color: 'var(--text-tertiary)',
    fontFamily: 'monospace',
    marginBottom: '2px',
  },
  description: {
    fontSize: '13px',
    fontWeight: '500',
    wordBreak: 'break-word',
    lineHeight: '1.4',
    color: 'var(--text)',
  },
  stats: {
    display: 'flex',
    gap: '12px',
    fontSize: '11px',
    color: 'var(--text-secondary)',
    flexShrink: 0,
  },
  stat: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'flex-end',
  },
  statLabel: {
    color: 'var(--text-muted)',
    fontSize: '9px',
    textTransform: 'uppercase',
  },
  statValue: {
    fontFamily: 'monospace',
    fontWeight: '500',
  },
  expandIcon: {
    color: 'var(--text-muted)',
    fontSize: '10px',
    marginLeft: '6px',
  },
  expandedContent: {
    padding: '10px 14px',
    background: 'var(--bg-card-expanded)',
    maxHeight: '900px',
    overflowY: 'auto',
  },
  sectionHeader: {
    fontSize: '10px',
    color: 'var(--text-tertiary)',
    textTransform: 'uppercase',
    marginBottom: '8px',
    fontWeight: '500',
  },
  example: {
    marginBottom: '8px',
    padding: '8px 10px',
    background: 'var(--bg-example)',
    borderRadius: '4px',
    border: '1px solid var(--border-light)',
  },
  exampleMeta: {
    fontSize: '10px',
    color: 'var(--text-muted)',
    marginBottom: '4px',
    fontFamily: 'monospace',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  proteinId: {
    color: 'var(--link)',
    fontWeight: '600',
  },
  annotation: {
    color: 'var(--text-secondary)',
    fontStyle: 'italic',
    marginLeft: '8px',
  },
  uniprotLink: {
    color: 'var(--link)',
    textDecoration: 'none',
    fontSize: '11px',
    marginLeft: '4px',
    opacity: 0.6,
  },
  noExamples: {
    color: 'var(--text-muted)',
    fontSize: '12px',
    fontStyle: 'italic',
  },
  structureGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '8px',
    marginTop: '12px',
  },
  structureHeader: {
    fontSize: '10px',
    color: 'var(--text-tertiary)',
    textTransform: 'uppercase',
    marginTop: '16px',
    marginBottom: '8px',
    fontWeight: '500',
  },
  alignBar: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '10px',
    color: 'var(--text-tertiary)',
  },
  alignLabel: {
    textTransform: 'uppercase',
    fontWeight: '500',
  },
  alignBtn: {
    padding: '2px 8px',
    border: '1px solid var(--border-input)',
    borderRadius: '3px',
    background: 'var(--bg-input)',
    cursor: 'pointer',
    fontSize: '10px',
    color: 'var(--text-secondary)',
  },
  alignBtnActive: {
    padding: '2px 8px',
    border: '1px solid var(--accent)',
    borderRadius: '3px',
    background: 'var(--bg-card)',
    cursor: 'pointer',
    fontSize: '10px',
    color: 'var(--text)',
    fontWeight: '600',
  },
  densityBar: {
    width: '50px',
    height: '3px',
    background: 'var(--density-bar-bg)',
    borderRadius: '2px',
    overflow: 'hidden',
    marginTop: '3px',
  },
  densityFill: {
    height: '100%',
    background: 'var(--accent)',
    borderRadius: '2px',
  },
}

const FeatureCard = forwardRef(function FeatureCard({ feature, isHighlighted, forceExpanded, onClick, loadExamples, vocabLogits, darkMode }, ref) {
  const [expanded, setExpanded] = useState(false)
  const [detailProtein, setDetailProtein] = useState(null)
  const [examples, setExamples] = useState([])
  const [loadingExamples, setLoadingExamples] = useState(false)
  const examplesCacheRef = useRef(null)
  const scrollGroupRef = useRef([])
  const [alignMode, setAlignMode] = useState('start')
  const [bandLimits, setBandLimits] = useState({})  // per-band: how many examples to show (expandable)
  const [hoveredToken, setHoveredToken] = useState(null)
  const [aiLabel, setAiLabel] = useState(null)
  const [aiLoading, setAiLoading] = useState(false)

  // If forceExpanded changes to true, expand the card
  useEffect(() => {
    if (forceExpanded) {
      setExpanded(true)
    }
  }, [forceExpanded])

  // Reset scroll group when card collapses or alignMode changes
  useEffect(() => {
    scrollGroupRef.current = []
  }, [expanded, alignMode])

  // Lazy-load examples from DuckDB when card is expanded
  useEffect(() => {
    if (!expanded || !loadExamples || examplesCacheRef.current) return
    let cancelled = false
    setLoadingExamples(true)
    loadExamples(feature.feature_id).then(result => {
      if (cancelled) return
      examplesCacheRef.current = result
      setExamples(result)
      setLoadingExamples(false)
    }).catch(err => {
      if (cancelled) return
      console.error('Error loading examples for feature', feature.feature_id, err)
      setLoadingExamples(false)
    })
    return () => { cancelled = true }
  }, [expanded, loadExamples, feature.feature_id])

  const freq = feature.activation_freq || 0
  const maxAct = feature.max_activation || 0
  const bestF1 = feature.best_f1 || 0
  const description = feature.description || `Feature ${feature.feature_id}`

  const handleClick = () => {
    const willExpand = !expanded
    setExpanded(willExpand)
    if (onClick) {
      onClick(feature.feature_id, willExpand)
    }
  }

  return (
    <div ref={ref} style={isHighlighted ? styles.cardHighlighted : styles.card}>
      <div style={styles.header} onClick={handleClick}>
        <div style={styles.headerLeft}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <div style={styles.featureId}>Feature #{feature.feature_id}</div>
            {bestF1 > 0 && (
              <span style={{
                fontSize: '9px',
                fontWeight: '600',
                padding: '1px 5px',
                borderRadius: '3px',
                background: bestF1 >= 0.5 ? 'rgba(118, 185, 0, 0.15)' : 'rgba(255, 165, 0, 0.15)',
                color: bestF1 >= 0.5 ? '#76b900' : '#ef9100',
                whiteSpace: 'nowrap',
              }}>
                F1: {bestF1.toFixed(2)}
              </span>
            )}
            <button
              title="Generate an autointerp label for this feature live (NIM)"
              onClick={async (e) => {
                e.stopPropagation(); setAiLoading(true); setAiLabel(null)
                const model = new URLSearchParams(window.location.search).get('model') || ''
                try {
                  const r = await fetch(`${window.location.protocol}//${window.location.hostname}:5199/autointerp`, {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, feature_id: feature.feature_id, bands: true }),
                  })
                  const d = await r.json(); setAiLabel(d)
                } catch (err) { setAiLabel({ error: '(autointerp server offline — tunnel/start port 5199)' }) }
                setAiLoading(false)
              }}
              style={{ fontSize: '10px', padding: '2px 8px', borderRadius: '4px', border: '1px solid #76b900',
                       background: 'transparent', color: '#76b900', cursor: 'pointer', whiteSpace: 'nowrap' }}
            >{aiLoading ? '⏳ interpreting…' : '🔍 Auto-interpret'}</button>
          </div>
          <div style={styles.description}>{description}</div>
          {aiLabel && (
            <div style={{ fontSize: '11px', marginTop: '4px' }}>
              {aiLabel.error ? <span style={{ color: '#e88' }}>{aiLabel.error}</span> : (<>
                <div style={{ color: '#bfe', whiteSpace: 'pre-wrap' }}>🔍 {aiLabel.label}</div>
                {aiLabel.shared_concept && (
                  <div style={{ fontSize: '11px', marginTop: '3px', padding: '2px 6px', borderRadius: '4px',
                    background: aiLabel.shared_concept.startsWith('SHARED') ? 'rgba(118,185,0,0.15)' : 'rgba(120,120,120,0.12)',
                    color: aiLabel.shared_concept.startsWith('SHARED') ? '#76b900' : '#9aa' }}>
                    🔗 {aiLabel.shared_concept}
                  </div>
                )}
                {aiLabel.protein_enrichment && aiLabel.protein_enrichment.length > 0 && (
                  <div style={{ fontSize: '11px', marginTop: '3px', padding: '2px 6px', borderRadius: '4px',
                    background: 'rgba(37,99,235,0.15)', color: '#93c5fd' }}>
                    🧬 protein-token GO: {aiLabel.protein_enrichment.map(e =>
                      `${e.name || e.go} (${e.k}/${e.n}, p=${e.p < 1e-4 ? e.p.toExponential(1) : e.p.toFixed(4)})`).join(' · ')}
                  </div>
                )}
                {aiLabel.uniprot_enrichment && (aiLabel.uniprot_enrichment.keywords?.length || aiLabel.uniprot_enrichment.domains?.length) ? (
                  <div style={{ fontSize: '11px', marginTop: '3px', padding: '2px 6px', borderRadius: '4px',
                    background: 'rgba(20,120,180,0.15)', color: '#7dd3fc' }}>
                    🔬 UniProt: {[...(aiLabel.uniprot_enrichment.keywords||[]), ...(aiLabel.uniprot_enrichment.domains||[])]
                      .sort((a,b)=>a.p-b.p).slice(0,4).map(e => `${e.term} (${e.k}/${e.n}, p=${e.p<1e-4?e.p.toExponential(1):e.p.toFixed(4)})`).join(' · ')}
                  </div>
                ) : null}
                {aiLabel.sample_enrichment && aiLabel.sample_enrichment.length > 0 && (
                  <div style={{ fontSize: '11px', marginTop: '3px', padding: '2px 6px', borderRadius: '4px',
                    background: 'rgba(139,92,246,0.15)', color: '#c4b5fd' }}>
                    🧩 sample-protein GO (what the samples are about): {aiLabel.sample_enrichment.map(e =>
                      `${e.name || e.go} (${e.k}/${e.n}, p=${e.p < 1e-4 ? e.p.toExponential(1) : e.p.toFixed(4)})`).join(' · ')}
                  </div>
                )}
                {aiLabel.peak_tokens && <div style={{ color: '#888', fontSize: '10px' }}>peak: {aiLabel.peak_tokens}</div>}
                {aiLabel.band_labels && Object.entries(aiLabel.band_labels).map(([b, v]) => {
                  const icon = { reasoning: '📖', answer: '✅', prompt: '❓', protein: '🧬', dna: '🧬', text: '📝', go: '🏷️' }[b] || '•'
                  return (
                    <div key={b} style={{ marginTop: '4px', paddingLeft: '6px', borderLeft: '2px solid #345' }}>
                      <div style={{ color: '#9cf', fontSize: '10px', fontWeight: 600 }}>
                        {icon} {b} <span style={{ color: '#678', fontWeight: 400 }}>(peak {v.peak_activation}, n={v.n_examples})</span>
                      </div>
                      <div style={{ color: '#bfe', whiteSpace: 'pre-wrap', fontSize: '10px' }}>{v.label}</div>
                    </div>
                  )
                })}
                {(aiLabel.windows || []).slice(0, 6).map((w, i) => (
                  <div key={i} style={{ fontFamily: 'monospace', color: '#9a9', fontSize: '10px', margin: '1px 0' }}>{w}</div>
                ))}
              </>)}
            </div>
          )}
        </div>
        <div style={styles.stats}>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Freq</span>
            <span style={styles.statValue}>{(freq * 100).toFixed(1)}%</span>
            <div style={styles.densityBar}>
              <div style={{ ...styles.densityFill, width: `${Math.min(freq * 100 * 10, 100)}%` }} />
            </div>
          </div>
          <div style={styles.stat}>
            <span style={styles.statLabel}>Max</span>
            <span style={styles.statValue}>{maxAct.toFixed(1)}</span>
          </div>
          <span style={styles.expandIcon}>{expanded ? '\u25BC' : '\u25B6'}</span>
        </div>
      </div>

      {expanded && (
        <div style={styles.expandedContent}>
          {/* Decoder Logits */}
          {vocabLogits && vocabLogits[String(feature.feature_id)] && (() => {
            const logits = vocabLogits[String(feature.feature_id)]
            const tokenLogitMap = {}
            for (const [tok, val] of (logits.top_positive || [])) tokenLogitMap[tok] = val
            for (const [tok, val] of (logits.top_negative || [])) tokenLogitMap[tok] = val
            // Show all 20 standard amino acids in alphabetical order
            const AMINO_ACIDS = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
            const allTokens = AMINO_ACIDS
            const maxAbs = Math.max(...AMINO_ACIDS.map(aa => Math.abs(tokenLogitMap[aa] || 0)), 0.001)
            return (
              <div style={{ marginBottom: '12px' }}>
                <div style={styles.sectionHeader}>Decoder Logits (mean-centered)</div>
                <div style={{ position: 'relative' }}>
                  {hoveredToken && (() => {
                    const val = tokenLogitMap[hoveredToken] || 0
                    return (
                      <div style={{
                        position: 'absolute', top: '-18px', left: '50%', transform: 'translateX(-50%)',
                        fontSize: '9px', fontFamily: 'monospace', fontWeight: '600',
                        color: 'var(--text)', background: 'var(--bg-card)',
                        border: '1px solid var(--border)', borderRadius: '3px',
                        padding: '1px 5px', whiteSpace: 'nowrap', zIndex: 1,
                        pointerEvents: 'none',
                      }}>
                        {hoveredToken}: {val > 0 ? '+' : ''}{val.toFixed(3)}
                      </div>
                    )
                  })()}
                  <div style={{ display: 'flex', width: '100%', gap: '1px', alignItems: 'flex-end', height: '32px' }}>
                    {allTokens.map(tok => {
                      const val = tokenLogitMap[tok] || 0
                      const h = Math.max(1, (Math.abs(val) / maxAbs) * 28)
                      const isHovered = hoveredToken === tok
                      const barColor = val === 0 ? 'var(--text-muted)' : val > 0 ? '#76b900' : '#e57373'
                      return (
                        <div key={tok} style={{
                          flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center',
                          justifyContent: 'flex-end', height: '32px', minWidth: 0,
                        }}
                          onMouseEnter={() => setHoveredToken(tok)}
                          onMouseLeave={() => setHoveredToken(null)}
                        >
                          <div style={{
                            width: '100%', maxWidth: '14px', height: `${h}px`, borderRadius: '1px',
                            background: barColor,
                            opacity: val === 0 ? 0.4 : Math.abs(val) / maxAbs * 0.7 + 0.3,
                            outline: isHovered ? '1.5px solid var(--text)' : 'none',
                            outlineOffset: '-0.5px',
                          }} />
                        </div>
                      )
                    })}
                  </div>
                  <div style={{ display: 'flex', width: '100%', gap: '1px', marginTop: '1px' }}>
                    {allTokens.map(tok => (
                      <div key={tok} style={{
                        flex: 1, textAlign: 'center', fontSize: '7px', fontWeight: '600',
                        color: 'var(--text-tertiary)', minWidth: 0, overflow: 'hidden',
                      }}>
                        {tok}
                      </div>
                    ))}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '10px', marginTop: '4px', fontSize: '9px', color: 'var(--text-tertiary)' }}>
                  <span><span style={{ display: 'inline-block', width: '8px', height: '8px', background: '#76b900', borderRadius: '1px', marginRight: '3px' }} />Promoted</span>
                  <span><span style={{ display: 'inline-block', width: '8px', height: '8px', background: '#e57373', borderRadius: '1px', marginRight: '3px' }} />Suppressed</span>
                  <span style={{ marginLeft: 'auto', fontStyle: 'italic' }}>relative to average feature</span>
                </div>
              </div>
            )
          })()}

          {/* Protein sequence examples */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <div style={styles.sectionHeader}>Top Activating Proteins</div>
            <div style={styles.alignBar}>
              <span style={styles.alignLabel}>Align by:</span>
              {['start', 'first_activation', 'max_activation'].map(mode => (
                <button
                  key={mode}
                  style={alignMode === mode ? styles.alignBtnActive : styles.alignBtn}
                  onClick={(e) => { e.stopPropagation(); setAlignMode(mode) }}
                >
                  {mode === 'start' ? 'sequence start' : mode === 'first_activation' ? 'first activation' : 'max activation'}
                </button>
              ))}
            </div>
          </div>
          {loadingExamples ? (
            <div style={{ textAlign: 'center', padding: '20px', color: 'var(--text-tertiary)', fontSize: '13px' }}>
              Loading examples...
            </div>
          ) : examples.length > 0 ? (
            <>
              {(() => {
                // Group examples BY BAND so cross-modal features show BOTH their protein and text parts.
                const BAND_META = {
                  dna:         { label: 'DNA',          color: '#0891b2' },
                  dna_ref:     { label: 'DNA (ref)',    color: '#0891b2' },
                  dna_variant: { label: 'DNA (variant)', color: '#e11d48' },
                  protein:   { label: 'PROTEIN',   color: '#2563eb' },
                  go:        { label: 'GO',        color: '#16a34a' },
                  prompt:    { label: 'PROMPT (question)', color: '#64748b' },
                  question:  { label: 'QUESTION',  color: '#64748b' },
                  reasoning: { label: 'REASONING', color: '#9333ea' },
                  answer:    { label: 'ANSWER',    color: '#ea580c' },
                  text:      { label: 'TEXT',      color: '#9333ea' },
                }
                const byBand = {}
                for (const ex of examples) {
                  const b = ex.band || 'text'
                  ;(byBand[b] = byBand[b] || []).push(ex)
                }
                // 'dna' + 'question' included so DNA features show their DNA-side and question-side activations
                const bandsPresent = ['dna_ref', 'dna_variant', 'dna', 'protein', 'go', 'prompt', 'question', 'reasoning', 'answer', 'text'].filter(b => byBand[b]?.length)
                const perBand = bandsPresent.length > 1 ? 5 : 10  // default shown per band; expandable below
                // peak activation per band + the overall top, so WEAK (spurious) bands can be flagged
                const bandPeak = b => Math.max(...byBand[b].map(e => e.max_activation || 0))
                const topPeak = Math.max(...bandsPresent.map(bandPeak), 1e-9)
                return bandsPresent.map(b => {
                  const meta = BAND_META[b] || { label: b.toUpperCase(), color: '#666' }
                  const limit = bandLimits[b] ?? perBand
                  const bandExamples = byBand[b].slice(0, limit)
                  const peak = bandPeak(b)
                  const weak = peak < 0.3 * topPeak  // fires <30% as strongly as the feature's dominant band
                  const { anchor: alignAnchor, totalLength } = computeAlignInfo(bandExamples, alignMode)
                  return (
                    <div key={b} style={weak ? { opacity: 0.5 } : undefined}>
                      <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.5, color: meta.color,
                                    borderLeft: `3px solid ${meta.color}`, paddingLeft: 6, margin: '8px 0 4px' }}>
                        {meta.label} examples ({byBand[b].length}) · peak {peak.toFixed(1)}
                        {weak && <span style={{ fontWeight: 400, fontStyle: 'italic', color: '#999' }}> — weak / likely spurious</span>}
                      </div>
                      <div style={bandExamples.length > 8 ? { maxHeight: 520, overflowY: 'auto', paddingRight: 4 } : undefined}>
                      {bandExamples.map((ex, i) => (
                        <div key={`${b}-${i}`} style={styles.example}>
                          <div style={styles.exampleMeta}>
                            <span>
                              <span style={styles.proteinId}>{ex.protein_id}</span>
                              <a
                                href={uniprotUrl(getAccession(ex.protein_id, ex.alphafold_id))}
                                target="_blank"
                                rel="noopener noreferrer"
                                style={styles.uniprotLink}
                                onClick={e => e.stopPropagation()}
                                title="View on UniProt"
                              >
                                ↗
                              </a>
                              {ex.best_annotation && (
                                <span style={styles.annotation}>{ex.best_annotation}</span>
                              )}
                            </span>
                            <span>max: {ex.max_activation?.toFixed(3) || 'N/A'}</span>
                          </div>
                          {ex.go_terms && (
                            <div style={{ fontSize: 10, color: '#16a34a', margin: '2px 0 3px', lineHeight: 1.3 }}
                                 title="GO terms found in this window (resolved from go-basic.obo)">
                              🟢 {ex.go_terms}
                            </div>
                          )}
                          <ProteinSequence
                            sequence={ex.sequence}
                            activations={ex.activations}
                            maxActivation={ex.max_activation}
                            alignMode={alignMode}
                            alignAnchor={alignAnchor}
                            totalLength={totalLength}
                            scrollGroupRef={scrollGroupRef}
                          />
                        </div>
                      ))}
                      </div>
                      {byBand[b].length > perBand && (
                        <button
                          onClick={e => {
                            e.stopPropagation()
                            setBandLimits(prev => ({
                              ...prev,
                              [b]: (prev[b] ?? perBand) >= byBand[b].length ? perBand : byBand[b].length,
                            }))
                          }}
                          style={{ fontSize: 11, margin: '2px 0 6px', padding: '2px 8px', cursor: 'pointer',
                                   background: 'transparent', border: `1px solid ${meta.color}`, color: meta.color,
                                   borderRadius: 4 }}
                        >
                          {limit >= byBand[b].length ? `show top ${perBand}` : `show all ${byBand[b].length}`}
                        </button>
                      )}
                    </div>
                  )
                })
              })()}

              {/* 2x3 Mol* structure grid */}
              <div style={styles.structureHeader}>3D Structures (AlphaFold)</div>
              <div style={styles.structureGrid}>
                {examples.slice(0, 6).map((ex, i) => (
                  <MolstarThumbnail
                    key={`${feature.feature_id}-${ex.protein_id}-${i}`}
                    proteinId={ex.protein_id}
                    alphafoldId={ex.alphafold_id}
                    sequence={ex.sequence}
                    activations={ex.activations}
                    maxActivation={ex.max_activation}
                    onExpand={() => setDetailProtein(ex)}
                    darkMode={darkMode}
                  />
                ))}
              </div>
            </>
          ) : (
            <div style={styles.noExamples}>No examples available</div>
          )}
        </div>
      )}

      {detailProtein && (
        <ProteinDetailModal
          protein={detailProtein}
          onClose={() => setDetailProtein(null)}
          darkMode={darkMode}
        />
      )}
    </div>
  )
})

export default FeatureCard
