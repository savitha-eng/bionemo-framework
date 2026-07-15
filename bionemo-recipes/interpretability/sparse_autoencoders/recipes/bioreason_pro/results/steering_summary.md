# Steering experiments — feature IDs + validated results

All **Layer 30**, SAE `sae-l30-exp16-balanced`, **reasoning-band** features. Clamp = set-activation on the
whole cluster during generation, held-out proteins, α sweep. "Coherent-injection rate" = fraction of
**zero-baseline** proteins (0 concept words at α=0) that gain the concept AND stay coherent by the
**char-level** metric (not the broken lexical one). See `steering_validation.json`, `scripts/analyze_traces.py`.

| cluster | clamped feature IDs (L30) | best feats (auc/coh) | coherent-injection @α180 | verdict |
|---|---|---|---|---|
| **synapse** | `16494, 4456, 22453, 21531, 39580, 30677` | F16494 0.99/0.55, F4456 0.98/0.8 | **0.77** (n=40) | ✅ steers coherently |
| **mito** | `30713, 10643, 20498, 36378, 15839, 24389` | F30713 0.95/0.95, F36378 0.98/0.65 | **0.60** (n=40) | ✅ steers coherently |
| **nervsys** | `4306, 6569, 4939, 29338, 3606, 11608` | F4306 0.94/0.75 | 0.36 (n=40) | ⚠️ moderate, degenerates @α210 |
| **transcription** | `21229, 24606, 12706, 22131, 29013, 37767` | F24606 0.94/0.5 | ~0.12 (1 protein) | ⚠️ weak / word-list overlap |
| **oxidoreductase** | `18980, 18389, 39935, 3310, 9189, 30488` | F39935 0.98/0.55 | 0.0 | ⚠️ one-protein baseline |
| **kinase** | `35647, 23666, 32179, 20129` | F35647 0.98/0.65 | 0.0 (present at α=0) | ❌ baseline-confounded |
| **ER** | `4716, 31592, 2955, 37876, 5362, 4185` | F4716 0.99/**1.0** | 0.0 | ❌ no injection (cleanest *detector*, not a handle) |
| **transporter** | `36696, 30993, 5687, 18556, 11908, 16271` | F36696 0.93/0.9 | 0.0 | ❌ no injection |

**Controls:** random matched-norm direction on the synapse feats → 0 injection everywhere (null).
**Injection onset:** α≈180 (α=90/135 inject essentially nothing); α≥210 raises injection but degeneration risk.

Trace files (full original-vs-clamped text): `results/traces_<cluster>.md`.

## LLM-judge (gold standard) — lexical injection vs genuine redirection

An independent LLM (llama-3.1-70b, NIM) rated each synapse generation on **concept genuinely present**
(not just sprinkled) AND **coherent**. On synapse (n≈13/α, NIM rate-limited):

| α | %genuine-concept | %coherent | %BOTH | char-metric rate |
|---|---|---|---|---|
| 0 | 0 | 100 | 0 | 0 |
| 180 | 31 | 92 | **31%** | 0.73 |
| 210 | 21 | 93 | 21% | 0.68 |

The LLM-judge %BOTH (**31%**) is ~half the char-metric rate (0.73). The gap is **sprinkled words vs
genuine reasoning**: the char-metric counts any coherent generation containing synapse words; the judge
requires the model to *genuinely reason about* synapse. So steering is **~73% coherent lexical injection
but only ~31% genuine reasoning-redirection** — mostly grafting concept vocabulary onto intact reasoning
(e.g. "binds post-synaptic density marks on histone H3"), genuinely redirecting ~1/3 of the time.

## Normalization fix (calibration) — result preserved

The SAE has `normalize_input=True`, so decoder directions live in normalized (per-token) space. Steering was
adding the delta to the raw residual *without* denormalizing by each token's std (direction right, magnitude
uncalibrated). Fixed (`h += delta * std`). Effect: the injection onset moved from α≈180 to **α≈28**, but the
char-level coherent-injection rate is **0.75 at α=28 (n=20)** — essentially identical to the old-scale 0.73.
So the bug only mis-scaled α; the finding reproduces.
