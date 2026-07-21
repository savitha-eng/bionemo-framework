# Reasoning-concept probe features (L30) — for dashboard lookup (?model=l30_balanced)

The L1-sparse linear probe per GO concept. **Only concepts that clear the leak floor (margin >=0.05) are
trustworthy** — the rest match a random projection (leakage). Feature IDs = the probe's nonzero coefficients.

| concept | trust | #feat | margin | top feature IDs (search these) |
|---|---|---|---|---|
| defense resp. to fungus | ✅ REAL | 12 | +0.111 | 36488, 35336, 2808, 23726, 29332, 32785, 7665, 15775, 22156, 2082, 28215, 5047 |
| defense resp. to bacterium | ✅ REAL | 36 | +0.108 | 35336, 38712, 36769, 8824, 2724, 32785, 20730, 22077, 13479, 25134, 27105, 10256 |
| structural molecule | ✅ REAL | 73 | +0.063 | 8476, 5782, 34525, 12813, 37792, 34509, 8663, 1142, 31342, 13347, 8095, 34403 |
| plasma membrane | ✅ REAL | 362 | +0.055 | 28961, 38367, 24751, 21955, 1256, 18533, 13140, 16971, 28374, 32738, 17108, 2627 |
| nucleus | ❌ leak | 362 | +0.035 | 2078, 28486, 21336, 316, 26706, 25277, 5310, 28374, 30127, 363, 7887, 39541 |
| catalytic (enzyme) | ❌ leak | 386 | +0.024 | 7152, 22120, 5682, 34367, 28475, 38208, 24674, 1639, 23666, 38010, 35001, 28 |
| sexual reproduction | ❌ leak | 106 | +0.021 | 1988, 21789, 23017, 4306, 7514, 31122, 28879, 6261, 25086, 18657, 20229, 11906 |
| reproduction | ❌ leak | 257 | +0.016 | 1988, 21789, 12100, 29548, 28879, 39286, 40425, 22358, 17565, 19966, 26741, 21486 |
| oxidoreductase | ❌ leak | 104 | +0.016 | 13676, 27248, 7322, 16952, 39935, 30488, 23433, 14029, 22120, 13126, 2102, 30595 |
| mitochondrion | ❌ leak | 159 | +0.013 | 36378, 2630, 670, 7443, 26232, 5001, 15477, 37996, 18058, 24751, 13062, 8780 |
| kinase activity | ❌ leak | 80 | +0.013 | 23666, 3787, 13277, 32338, 20129, 23554, 15458, 15262, 158, 26032, 7136, 20424 |
| transporter activity | ❌ leak | 96 | +0.006 | 18211, 36696, 516, 16271, 31418, 39402, 20500, 14529, 7152, 30993, 15659, 6445 |
