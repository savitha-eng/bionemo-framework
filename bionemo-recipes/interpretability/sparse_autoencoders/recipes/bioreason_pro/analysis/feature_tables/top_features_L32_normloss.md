# BioReason-Pro SAE — top interpretable features (L32, normalize_loss)

Each SAE feature characterized on the held-out 300-protein eval set. `go_label` = the GO concept
it best predicts (held-out de-biased per-protein AUC). `band_class` = which modality it fires on
(by per-band fire RATE). `fusion_class` = truly cross-modal (text-involving SAE-V fusion) or unimodal.
Full table: features_L32_normloss.csv (all features).

| feat | GO concept | AUC | band | fusion | fire% | omega |
|---|---|---|---|---|---|---|
| 12995 | cytosol | 0.9752 | text-heavy | unimodal | 0.08 | 0.0 |
| 8489 | cytosol | 0.9712 | text-heavy | unimodal | 0.12 | 0.0 |
| 12734 | nucleus | 0.9646 | text-heavy | unimodal | 47.67 | 0.0 |
| 5567 | positive regulation of biological process | 0.9598 | text-heavy | unimodal | 0.16 | 0.0 |
| 8431 | positive regulation of biological process | 0.9562 | text-heavy | unimodal | 0.15 | 0.0 |
| 17454 | cytosol | 0.9544 | text-heavy | unimodal | 0.18 | 0.0 |
| 10318 | plasma membrane | 0.9503 | text-heavy | unimodal | 0.12 | 0.0 |
| 13476 | protein-containing complex | 0.9497 | text-heavy | unimodal | 0.08 | 0.0 |
| 3052 | cytosol | 0.9425 | text-heavy | unimodal | 0.06 | 0.0 |
| 14106 | nucleic acid binding | 0.938 | text-heavy | unimodal | 0.11 | 0.0 |
| 11599 | anatomical structure development | 0.9358 | text-heavy | unimodal | 1.44 | 0.0 |
| 2743 | catalytic activity | 0.9334 | text-heavy | unimodal | 0.39 | 0.0 |
| 18897 | multicellular organism development | 0.9328 | text-heavy | unimodal | 0.15 | 0.0 |
| 659 | cytosol | 0.9312 | text-heavy | unimodal | 0.1 | 0.0 |
| 6560 | cytosol | 0.9292 | text-heavy | unimodal | 0.15 | 0.0 |
| 8267 | nucleus | 0.9271 | text-heavy | unimodal | 0.27 | 0.0 |
| 16847 | multicellular organism development | 0.9244 | text-heavy | unimodal | 0.22 | 0.0 |
| 16910 | positive regulation of biological process | 0.9237 | text-heavy | unimodal | 0.16 | 0.0 |
| 5046 | negative regulation of biological process | 0.9228 | text-heavy | unimodal | 0.23 | 0.0 |
| 15803 | anatomical structure development | 0.9223 | text-heavy | unimodal | 0.21 | 0.0 |
| 2943 | positive regulation of biological process | 0.9221 | text-heavy | unimodal | 1.07 | 0.0 |
| 8675 | cytosol | 0.9216 | text-heavy | unimodal | 0.06 | 0.0 |
| 14939 | nucleic acid binding | 0.9212 | text-heavy | unimodal | 0.18 | 0.0 |
| 19573 | catalytic activity | 0.9203 | text-heavy | unimodal | 0.61 | 0.0 |
| 15053 | catalytic activity | 0.92 | text-heavy | unimodal | 0.07 | 0.0 |
| 4618 | protein-containing complex | 0.9199 | text-heavy | unimodal | 0.33 | 0.0 |
| 11794 | nucleic acid binding | 0.9191 | text-heavy | unimodal | 0.21 | 0.0 |
| 1275 | cytosol | 0.9183 | text-heavy | unimodal | 0.13 | 0.0 |
| 19627 | plasma membrane | 0.9158 | text-heavy | unimodal | 0.05 | 0.0 |
| 7758 | cytosol | 0.9144 | text-heavy | unimodal | 0.13 | 0.0 |
| 9960 | plasma membrane | 0.9137 | text-heavy | unimodal | 0.1 | 0.0 |
| 13674 | plasma membrane | 0.9133 | text-heavy | unimodal | 0.03 | 0.0 |
| 17831 | nucleus | 0.912 | text-heavy | unimodal | 0.09 | 0.0 |
| 11156 | nucleic acid binding | 0.9118 | text-heavy | unimodal | 0.25 | 0.0 |
| 5779 | multicellular organism development | 0.9107 | text-heavy | unimodal | 0.3 | 0.0 |
| 19767 | cytosol | 0.9104 | text-heavy | unimodal | 0.08 | 0.0 |
| 156 | catalytic activity | 0.9101 | text-heavy | unimodal | 0.06 | 0.0 |
| 15524 | catalytic activity | 0.9094 | text-heavy | unimodal | 0.04 | 0.0 |
| 18275 | cytosol | 0.9094 | text-heavy | unimodal | 0.13 | 0.0 |
| 17849 | nucleic acid binding | 0.9081 | text-heavy | unimodal | 0.03 | 0.0 |
| 16763 | nucleic acid binding | 0.9076 | text-heavy | unimodal | 0.23 | 0.0 |
| 19022 | nucleic acid binding | 0.9076 | text-heavy | unimodal | 0.24 | 0.0 |
| 14452 | catalytic activity | 0.9072 | text-heavy | unimodal | 0.04 | 0.0 |
| 9178 | catalytic activity | 0.9072 | text-heavy | unimodal | 0.43 | 0.0 |
| 1659 | cytosol | 0.9064 | text-heavy | unimodal | 0.22 | 0.0 |
| 5701 | nucleic acid binding | 0.9049 | text-heavy | unimodal | 0.56 | 0.0 |
| 18054 | catalytic activity | 0.9043 | text-heavy | unimodal | 0.12 | 0.0 |
| 6165 | biosynthetic process | 0.904 | text-heavy | unimodal | 0.77 | 0.0 |
| 18508 | multicellular organism development | 0.9034 | text-heavy | unimodal | 0.39 | 0.0 |
| 5038 | cytosol | 0.9031 | text-heavy | unimodal | 0.09 | 0.0 |
| 19639 | nucleic acid binding | 0.9028 | text-heavy | unimodal | 0.98 | 0.0 |
| 6207 | response to stress | 0.9028 | text-heavy | unimodal | 0.32 | 0.0 |
| 13772 | nucleus | 0.9027 | text-heavy | unimodal | 0.19 | 0.0 |
| 707 | cytosol | 0.9024 | text-heavy | unimodal | 0.08 | 0.0 |
| 369 | localization | 0.902 | text-heavy | unimodal | 0.05 | 0.0 |
| 6710 | multicellular organism development | 0.9018 | text-heavy | unimodal | 0.5 | 0.0 |
| 13584 | nucleic acid binding | 0.9013 | text-heavy | unimodal | 0.13 | 0.0 |
| 11449 | multicellular organism development | 0.9013 | text-heavy | unimodal | 0.31 | 0.0 |
| 1797 | intracellular non-membrane-bounded organelle | 0.9009 | text-heavy | unimodal | 0.06 | 0.0 |
| 5810 | multicellular organism development | 0.8997 | text-heavy | unimodal | 0.2 | 0.0 |
| 18419 | multicellular organism development | 0.8997 | text-heavy | unimodal | 0.12 | 0.0 |
| 3459 | negative regulation of biological process | 0.8996 | text-heavy | unimodal | 0.4 | 0.0 |
| 13541 | plasma membrane | 0.8994 | text-heavy | unimodal | 1.08 | 0.0 |
| 16902 | nucleic acid binding | 0.8986 | text-heavy | unimodal | 0.09 | 0.0 |
| 9501 | multicellular organism development | 0.8986 | text-heavy | unimodal | 0.13 | 0.0 |
| 19074 | anatomical structure development | 0.8985 | text-heavy | unimodal | 0.31 | 0.0 |
| 8239 | nucleus | 0.8979 | text-heavy | unimodal | 0.31 | 0.0 |
| 14517 | response to stimulus | 0.8973 | text-heavy | unimodal | 0.12 | 0.0 |
| 2360 | catalytic activity | 0.896 | text-heavy | unimodal | 0.12 | 0.0 |
| 12020 | nucleic acid binding | 0.896 | text-heavy | unimodal | 0.08 | 0.0 |
| 5399 | multicellular organism development | 0.896 | text-heavy | unimodal | 0.24 | 0.0 |
| 13130 | multicellular organism development | 0.8955 | text-heavy | unimodal | 0.28 | 0.0 |
| 15776 | nucleic acid binding | 0.8955 | text-heavy | unimodal | 0.39 | 0.0 |
| 8059 | multicellular organism development | 0.8944 | text-heavy | unimodal | 0.17 | 0.0 |
| 17729 | nucleic acid binding | 0.8944 | text-heavy | unimodal | 0.22 | 0.0 |
| 6152 | multicellular organism development | 0.8944 | text-heavy | unimodal | 0.19 | 0.0 |
| 7249 | nucleic acid binding | 0.8944 | text-heavy | unimodal | 0.18 | 0.0 |
| 12931 | cytosol | 0.8935 | text-heavy | unimodal | 0.13 | 0.0 |
| 18736 | plasma membrane | 0.8927 | text-heavy | unimodal | 0.04 | 0.0 |
| 11071 | developmental process | 0.8923 | text-heavy | unimodal | 2.77 | 0.0 |
| 14140 | developmental process | 0.8905 | text-heavy | unimodal | 0.22 | 0.0 |
| 9925 | organelle lumen | 0.8904 | text-heavy | unimodal | 0.16 | 0.0 |
| 17793 | nucleus | 0.8897 | text-heavy | unimodal | 0.04 | 0.0 |
| 3702 | nucleus | 0.8895 | text-heavy | unimodal | 0.64 | 0.0 |
| 3039 | nucleic acid binding | 0.8892 | text-heavy | unimodal | 0.34 | 0.0 |
| 15872 | multicellular organism development | 0.8892 | text-heavy | unimodal | 0.18 | 0.0 |
| 3286 | catalytic activity | 0.889 | text-heavy | unimodal | 0.24 | 0.0 |
| 10788 | nucleic acid binding | 0.8881 | text-heavy | unimodal | 0.35 | 0.0 |
| 17332 | nucleic acid binding | 0.8881 | text-heavy | unimodal | 0.16 | 0.0 |
| 7179 | organelle lumen | 0.8877 | text-heavy | unimodal | 0.04 | 0.0 |
| 10956 | positive regulation of biological process | 0.8867 | text-heavy | unimodal | 0.15 | 0.0 |
| 20361 | catalytic activity | 0.8861 | text-heavy | unimodal | 0.16 | 0.0 |
| 13799 | multicellular organism development | 0.886 | text-heavy | unimodal | 0.53 | 0.0 |
| 17526 | multicellular organismal process | 0.8854 | text-heavy | unimodal | 0.07 | 0.0 |
| 5159 | negative regulation of biological process | 0.8852 | text-heavy | unimodal | 0.1 | 0.0 |
| 11987 | nucleic acid binding | 0.885 | text-heavy | unimodal | 0.5 | 0.0 |
| 2702 | plasma membrane | 0.8834 | text-heavy | unimodal | 0.08 | 0.0 |
| 15357 | plasma membrane | 0.8834 | text-heavy | unimodal | 0.13 | 0.0 |
| 20280 | multicellular organism development | 0.8824 | text-heavy | unimodal | 0.25 | 0.0 |
| 755 | plasma membrane | 0.8822 | text-heavy | unimodal | 0.17 | 0.0 |
