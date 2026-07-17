#!/usr/bin/env python
"""Cryo-EM inter-chain contact residues for the paper's §2.7 case studies.
eEFSec (P57772) in 7ZJW (bound to SECIS RNA); CFAP61 (Q8NHU2) in 8J07 (bound to dynein DHC1).
For each: identify the target protein chain, find its residues within 5A of any OTHER chain (the partner
interface), and report them. Confirms the repurposed residues C734/V737/V739 for CFAP61.
"""
import sys, urllib.request, numpy as np
import biotite.structure.io.pdbx as pdbx, biotite.structure as struc
from biotite.database.rcsb import fetch

def contacts(pdb_id, target_len_hint):
    path = fetch(pdb_id, "cif", "/data/savithas/phase3_full/cryoem")
    arr = pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    prot = arr[struc.filter_amino_acids(arr)]
    chains = {}
    for ch in np.unique(prot.chain_id):
        c = prot[prot.chain_id == ch]
        chains[ch] = len(np.unique(c.res_id))
    print(f"\n=== {pdb_id} === protein chains (chain: #residues): {chains}")
    # target = the amino-acid chain whose length best matches the target protein
    tgt = min(chains, key=lambda c: abs(chains[c] - target_len_hint))
    print(f"  target chain guess: {tgt} ({chains[tgt]} res, hint {target_len_hint})")
    tprot = prot[prot.chain_id == tgt]
    # partner = ALL other atoms (protein of other chains + any non-protein like RNA)
    other = arr[arr.chain_id != tgt]
    tc = tprot.coord; oc = other.coord
    # per target residue: min distance to any partner atom
    contact_res = []
    for rid in np.unique(tprot.res_id):
        rc = tprot[tprot.res_id == rid].coord
        d = np.sqrt(((rc[:, None] - oc[None]) ** 2).sum(-1)).min()
        if d < 5.0:
            nm = tprot[tprot.res_id == rid].res_name[0]
            contact_res.append((int(rid), nm))
    print(f"  contact residues (<5A of a partner chain): {len(contact_res)}")
    print(f"  positions: {[r for r,_ in contact_res][:40]}")
    return tgt, contact_res

_, c8j07 = contacts("8J07", 1237)   # CFAP61
print("  CFAP61 repurposed active-site residues C734/V737/V739 among contacts?",
      [r for r,_ in c8j07 if r in (734,737,739)])
_, c7zjw = contacts("7ZJW", 596)    # eEFSec (~596 aa)
