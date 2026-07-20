# Layer Robustness Check — L16 / L22 vs L30

_Are the L30 conclusions layer-robust, or a late-layer artifact? Re-ran the core analyses at a middle (L22, ~61%
depth) and early (L16, ~44%) layer, same 8k store (apples-to-apples). Model is 36 layers; L30 ~83%._
**⚠️ All on the TRAIN split (the only activation stores that exist). Protein-band = ESM3-sourced (frozen → train
membership minor); reasoning-band inherits the LLM-SFT memorization caveat on top of leakage.**

## Fixes carried (audited after user challenge)
`probe_v2` (same script as L30): SAE-svd vs raw vs random, dim-matched, reasoning=response — ✅.
`layer_robustness.py`: **winner's-curse train/test split** (select feature on train, report test) — ✅ added.
Freq/sink filter: per-protein firing-rate proxy (documented; real localization = domain-F1).

## Result 1 — Decodability (probe_v2): both theses LAYER-ROBUST
| band | metric | L16 | L22 | L30 |
|---|---|---|---|---|
| protein | SAE / raw / random | 0.807 / 0.807 / 0.797 | 0.807 / 0.807 / 0.795 | SAE=raw |
| reasoning | SAE / raw / random | 0.939 / 0.946 / **0.949** | 0.948 / 0.951 / **0.956** | 0.938 / 0.954 / 0.946 |

- **SAE never beats raw on the protein band** — at every layer. Not a late-layer artifact.
- **Reasoning-band GO is leaked at every layer** — at L16/L22 the *random* projection scores **highest** (leakage even starker mid-network). (Dense probe; sparse-beats-raw is an L30 `probe_trained` result, not re-run per layer.)

## Result 2 — Multimodal activity is NOT higher mid-network (validates L30)
| | L16 | L22 | L30 |
|---|---|---|---|
| reasoning mean-active (median) | 0.011 | 0.011 | **0.035** |
| features active in BOTH bands (Sense-B) | ~32 | ~24 | ~32 |
| cross-modal pairing r≥0.4 | — | 15/15 (r 0.67) | 20/20 (r 0.70) |

- **Reasoning features are ~3× weaker at L16/L22** — the reasoning representation builds up with depth; **L30 is the reasoning sweet spot.**
- **Same-feature fusion is tiny everywhere (~24–32 / 40,960) and NOT higher mid-network** — no fusion surge; "no feature-level fusion" is layer-robust.
- **Cross-feature alignment holds** (15/15 at L22, concept-matched: GPCR F4298→F21938 r=0.77, Ig F2458→F794 0.71, P-loop F5339→F26251 0.71) — same convergence pattern as L30.

## Consequence
The L22 2×2 causal test would be **underpowered** (reasoning cluster mean-active ~0.07, near-unsteerable) → a
null-on-null. The fusion question is already answered by Result 2 (no surge). **domain-F1 at L16/L22 (structure
localization near ESM3) is the remaining valuable check** — pending.

_Scripts: `layer_robustness.py`, `probe_v2.py`. Data: `layer_robustness_l{16,22}.json`, `probe_v2_*_l{16,22}.json`._

## Result 3 — Structure IS more localized early (domain-F1) — a structure/reasoning TRADE-OFF
Domain-F1 (per-position precision × per-region recall, Polina), best-feature-per-domain, apples-to-apples:
| layer | median domain-F1 | ≥0.5 localized | ≥0.7 |
|---|---|---|---|
| **L16 (early)** | 0.51 | **50%** | 25% |
| **L22 (middle)** | 0.52 | **56%** | 26% |
| L30 (late) | 0.30 | 32% | 17% |

**Structure features are ~2× more localized at L16/L22 than L30** — ESM3 residue structure is cleanest near the
input and mixes with depth. L30 UNDERSOLD structural interpretability. (recall ~0.97, precision ~0.35: fires in
every domain region but also somewhat outside.)

**THE LAYER TRADE-OFF:** structure localization best early (L16–L22, >50%), reasoning strength best late
(L30, steerable). No single layer optimizes both — structure lives near ESM3, reasoning builds up deep. For
structural interpretability use L16–L22 (earlier may be even better); for reasoning/steering use L30. **L32 helps
neither** (later = less localized structure + reasoning already declining, steer 0%).
