# Cross-modal feature pairs (L30)

Each row: a **protein/residue-band bio feature** and the **reasoning-band feature** it co-fires with across
proteins (Pearson r, permutation-null 0.07–0.25). **Sense A (cross-FEATURE) alignment** — different features,
same concept. NOT same-feature fusion (that null holds), NOT causal (2x2 showed structure is inert). Trust
pairs where the bio side is also LOCALIZED (domain-F1 >= 0.7); GPCR pairs have high r but weak bio localization.

| bio feat | domain-F1 | reasoning feat | r | bio concept | reasoning label | trust |
|---|---|---|---|---|---|---|
| F7369 | 0.51 | F3184 | 0.89 | G protein-coupled receptor | Transmembrane Domain | ~ok |
| F3623 | 0.49 | F13384 | 0.87 | Cytochrome P450 | Residue Range | ⚠️ weak bio-side |
| F13950 | 0.95 | F10235 | 0.86 | RNA-binding domain superfa | Zinc Finger Preceding RRM | ✅ both localized |
| F11025 | 0.73 | F10235 | 0.86 | RNA-binding domain superfa | Zinc Finger Preceding RRM | ✅ both localized |
| F17203 | 0.33 | F3184 | 0.83 | G protein-coupled receptor | Transmembrane Domain | ⚠️ weak bio-side |
| F35983 | 0.22 | F3184 | 0.82 | G protein-coupled receptor | Transmembrane Domain | ⚠️ weak bio-side |
| F36890 | 0.70 | F10235 | 0.82 | RNA-binding domain superfa | Zinc Finger Preceding RRM | ✅ both localized |
| F32546 | 0.28 | F3184 | 0.81 | G protein-coupled receptor | Transmembrane Domain | ⚠️ weak bio-side |
| F16964 | 0.80 | F4783 | 0.81 | Leucine-rich repeat domain | Leucine-rich repeat | ✅ both localized |
| F18162 | 0.93 | F6209 | 0.74 | Helix-loop-helix DNA-bindi | Residue Range | ✅ both localized |
| F36276 | 0.28 | F13384 | 0.74 | Cytochrome P450 | Residue Range | ⚠️ weak bio-side |
| F14532 | 0.89 | F29088 | 0.71 | Collagen triple helix repe | Collagen Binding | ✅ both localized |
| F4888 | 0.92 | F29088 | 0.71 | Collagen triple helix repe | Collagen Binding | ✅ both localized |
| F11836 | 0.63 | F21642 | 0.65 | Histone-fold | H2A-H2B Dimer Formation | ~ok |
| F33072 | 0.95 | F15673 | 0.63 | Kinesin motor domain super | Motor Function | ✅ both localized |
| F18393 | 0.98 | F15673 | 0.62 | Kinesin motor domain super | Motor Function | ✅ both localized |
| F4994 | 0.73 | F15673 | 0.54 | Kinesin motor domain super | Motor Function | ✅ both localized |
| F13025 | 0.71 | F17771 | 0.51 | C2 domain superfamily | PH domain location | ✅ both localized |
| F32701 | 0.42 | F40206 | 0.49 | Helicase, C-terminal domai | ATP-binding domain | ⚠️ weak bio-side |
| F21549 | 0.11 | F34713 | 0.41 | Cystine-knot cytokine | Growth Factor Binding | ⚠️ weak bio-side |

**Pattern:** multiple bio features for one fold converge on ONE reasoning partner (4 GPCR→F3184,
4 kinesin→F15673, 3 RNA-binding→F10235) — the SAE represents each concept with parallel features per
modality. Source: , .

Source: `crossmodal_localization_l30.json`, `crossmodal_pairing_l30.json`.
