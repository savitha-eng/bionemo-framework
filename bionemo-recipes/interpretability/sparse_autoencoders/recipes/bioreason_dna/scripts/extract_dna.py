r"""Extract BioReason-DNA residual-stream activations for SAE training (faithful fused forward).

Mirrors ``bioreason_pro/scripts/extract.py`` for the DNA variant-effect model. Runs the model's
real teacher-forced forward (precomputed Evo2-1B DNA embeddings projected through the trained
2-layer ``DNAProjection`` MLP + learned ``<|DNA_START|>``/``<|DNA_END|>`` marker embeddings spliced
into the placeholder slots), captures the residual stream at one or more LLM layers via a forward
hook on ``text_model.model.layers[L]``, drops pad tokens, and writes one SAE ``ActivationStore`` per
layer plus a row-aligned per-token band sidecar (dna / text) and a per-example metadata table.

Model / data provenance (studied from the authors' code, NOT the toy geometry script):
  - Checkpoint ``ev-4b-vep`` was produced by the ``bioreason_dna`` stack (branch
    ``savitha/dna-30b-stack``), NOT ``bioreason/models/dna_llm.py``. The saved
    ``dna_projection.pt`` is a 2-layer MLP (keys ``mlp.0`` 1920->2560, ``mlp.2`` 2560->2560), and
    ``dna_marker_embeddings.pt`` holds learned ``dna_start_embedding`` / ``dna_end_embedding``
    (2560-d) that replace the START/END token embeddings. The prior toy script FAILED because it
    fed raw sequences through the processor (0 DNA tokens) and loaded the projection into a single
    ``nn.Linear``. This script uses the real pipeline instead.
  - DNA is consumed as PRECOMPUTED Evo2-1B ``blocks.20.mlp.l3`` embeddings (dim 1920) from an
    ``EmbeddingStore``, keyed by ``sequence_id`` per role (reference, variant). The collator lays
    down ``<|DNA_START|> <|dna_pad|>*n <|DNA_END|>`` spans (2 per example) and the concatenated
    per-row embeddings scatter into the pad slots. Evo2 is never run here.

Single-GPU smoke (Step A, ~4-8 examples):
    CUDA_VISIBLE_DEVICES=<free_gpu> \
    /data/savithas/repos/bioreason-nemotron/.venv/bin/python scripts/extract_dna.py \
        --num-examples 6 --layers 16 --batch-size 2 --smoke \
        --output /data/savithas/dna_sae/activations/smoke6

Full extraction (all 36,088 sequences, single GPU — submit as a Lepton batch job):
    CUDA_VISIBLE_DEVICES=<gpu> \
    /data/savithas/repos/bioreason-nemotron/.venv/bin/python scripts/extract_dna.py \
        --num-examples 36088 --split all --layers 16 \
        --output /data/savithas/dna_sae/activations/vep_non_snv_L16_full
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Repo wiring -------------------------------------------------------------------------------
# The DNA data/model pieces (bioreason_dna.*) live on the savitha/dna-30b-stack branch; a detached
# worktree of that branch is checked out here so we can import them without disturbing the running
# training's checked-out branch. The SAE ActivationStore comes from this recipe tree.
DEFAULT_DNA_STACK = "/data/savithas/phase3_full/dna_extract_wt"
_SAE_SRC = Path(__file__).resolve().parents[3] / "sae" / "src"
sys.path.insert(0, str(_SAE_SRC))

DEFAULT_CKPT = "/data/savithas/converted/ev-4b-vep"
DEFAULT_EMB = "/data/savithas/dna_embeddings/evo2_1b/vep_non_snv"
DEFAULT_DATASET = "wanglab/variant_effect_non_snv"
DEFAULT_DATASET_NAME = "variant_effect_non_snv"  # basename used to key sequence_ids in the store


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Extract BioReason-DNA activations for SAE training")
    p.add_argument("--output", required=True,
                   help="Output dir; per-layer ActivationStores go under layer<L>/")
    p.add_argument("--ckpt-dir", default=DEFAULT_CKPT)
    p.add_argument("--dna-stack", default=DEFAULT_DNA_STACK,
                   help="Path to bioreason_dna importable tree (savitha/dna-30b-stack worktree)")
    p.add_argument("--embedding-base", default=DEFAULT_EMB,
                   help="EmbeddingStore base dir (precomputed Evo2-1B reference+variant embeddings)")
    p.add_argument("--dataset-path", default=DEFAULT_DATASET, help="HF dataset repo id")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME,
                   help="Basename that keyed sequence_ids at precompute time (used to make ids)")
    p.add_argument("--embedding-block", default="blocks.20.mlp.l3")
    p.add_argument("--embedding-prefix", default="evo2_1b")
    p.add_argument("--layers", type=int, nargs="+", default=[16],
                   help="Decoder layer indices to hook (output = residual stream after the block)")
    p.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    p.add_argument("--num-examples", type=int, default=6)
    p.add_argument("--dna-subsample", type=int, default=512,
                   help="Keep at most N randomly-sampled <|dna_pad|> token activations per example "
                        "(all text tokens + DNA START/END markers are always kept). <=0 disables.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for DNA-token subsampling")
    p.add_argument("--truncate-dna-per-side", type=int, default=1024,
                   help="Must match precompute (1024 -> 2048 nt/seq); embedding lengths depend on it")
    p.add_argument("--max-length-text", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--shard-size", type=int, default=200_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn-impl", default="flash_attention_2",
                   help="Attention impl for the Qwen3 text model (sdpa if flash-attn unavailable)")
    p.add_argument("--smoke", action="store_true",
                   help="Compute + print layer geometry (dna_self / text_self / dna<->text cross)")
    p.add_argument("--report-json", default=None)
    return p.parse_args()


# --- SAE store helpers (ported verbatim from bioreason_pro/scripts/extract.py) ------------------
class ShardAccumulator:
    """Buffer activations and hand the store exactly one shard_size block at a time (O(1) appends)."""

    def __init__(self, store, shard_size: int, executor=None, max_pending: int = 2):
        self.store = store
        self.shard_size = shard_size
        self.executor = executor
        self.max_pending = max_pending
        self._chunks = []
        self._rows = 0
        self._futures = []

    def add(self, arr: np.ndarray):
        self._chunks.append(arr)
        self._rows += arr.shape[0]
        if self._rows >= self.shard_size:
            self._drain()

    def _persist(self, block: np.ndarray):
        if self.executor is None:
            self.store.append(block)
            return
        while len(self._futures) >= self.max_pending:
            self._futures.pop(0).result()
        self._futures.append(self.executor.submit(self.store.append, block))

    def _drain(self):
        buf = np.concatenate(self._chunks, axis=0)
        n_full = (buf.shape[0] // self.shard_size) * self.shard_size
        if n_full:
            self._persist(np.ascontiguousarray(buf[:n_full]))
        rem = buf[n_full:]
        self._chunks = [rem.copy()] if rem.shape[0] else []
        self._rows = rem.shape[0]

    def close(self):
        if self._chunks:
            self._persist(np.ascontiguousarray(np.concatenate(self._chunks, axis=0)))
        self._chunks = []
        self._rows = 0
        for f in self._futures:
            f.result()
        self._futures = []


class LabelSidecar:
    """Row-aligned per-token band labels, flushed incrementally to one parquet."""

    _SCHEMA = pa.schema([
        ("sequence_id", pa.string()),
        ("token_index", pa.int32()),
        ("position_type", pa.string()),  # dna | text
    ])

    def __init__(self, path: Path, flush_rows: int = 1_000_000):
        self.writer = pq.ParquetWriter(str(path), self._SCHEMA, compression="snappy")
        self.flush_rows = flush_rows
        self._sid, self._idx, self._pt = [], [], []

    def append(self, sequence_id, token_indices, position_types):
        self._sid.extend([sequence_id] * len(token_indices))
        self._idx.extend(int(i) for i in token_indices)
        self._pt.extend(str(t) for t in position_types)
        if len(self._sid) >= self.flush_rows:
            self._flush()

    def _flush(self):
        if not self._sid:
            return
        batch = pa.record_batch(
            [pa.array(self._sid, pa.string()), pa.array(self._idx, pa.int32()), pa.array(self._pt, pa.string())],
            schema=self._SCHEMA,
        )
        self.writer.write_batch(batch)
        self._sid, self._idx, self._pt = [], [], []

    def close(self):
        self._flush()
        self.writer.close()


# --- Model loading (faithful reconstruction of BioReasonDNA's forward, no unsloth/evo2) ---------
class DNAExtractModel:
    """Minimal, hookable reconstruction of the trained BioReason-DNA forward.

    We do NOT instantiate ``bioreason_dna.models.llm.BioReasonDNA`` (it pulls unsloth + the
    multimodal loader path and re-derives LoRA/quant). Instead we load exactly what ``convert_ckpt``
    saved into ``ev-4b-vep``:
      - the LoRA-merged Qwen3-4B text model (``model.safetensors``) via AutoModelForCausalLM,
      - the 2-layer ``DNAProjection`` MLP (``dna_projection.pt``),
      - the learned marker embeddings (``dna_marker_embeddings.pt``),
    and replay ``BioReasonDNA.inject_dna_embeddings`` verbatim so the residual stream matches
    training.
    """

    def __init__(self, ckpt_dir: str, dna_stack: str, device: str, attn_impl: str,
                 dna_embedding_dim: int = 1920):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        sys.path.insert(0, dna_stack)
        from bioreason_dna.models.projection import DNAProjection
        from bioreason_dna.models.constants import (
            DNA_START_TOKEN, DNA_PAD_TOKEN, DNA_END_TOKEN, NUM_DNA_SEQUENCES_PER_EXAMPLE,
        )

        self.device = device
        self.num_dna_sequences = NUM_DNA_SEQUENCES_PER_EXAMPLE

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        self.text_model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
            trust_remote_code=True,
        ).to(device).eval()
        self.text_config = self.text_model.config
        hidden = self.text_config.hidden_size

        self.dna_start_token_id = self.tokenizer.convert_tokens_to_ids(DNA_START_TOKEN)
        self.dna_pad_token_id = self.tokenizer.convert_tokens_to_ids(DNA_PAD_TOKEN)
        self.dna_end_token_id = self.tokenizer.convert_tokens_to_ids(DNA_END_TOKEN)

        # 2-layer projection MLP (Linear-GELU-Linear); keys mlp.0 / mlp.2.
        self.dna_projection = DNAProjection(embedding_dim=dna_embedding_dim, output_dim=hidden, num_layers=2)
        proj_sd = torch.load(os.path.join(ckpt_dir, "dna_projection.pt"), map_location="cpu", weights_only=False)
        self.dna_projection.load_state_dict(proj_sd, strict=True)
        self.dna_projection = self.dna_projection.to(device=device, dtype=torch.bfloat16).eval()

        marker_sd = torch.load(os.path.join(ckpt_dir, "dna_marker_embeddings.pt"),
                               map_location="cpu", weights_only=False)
        self.dna_start_embedding = marker_sd["dna_start_embedding"].to(device=device, dtype=torch.bfloat16)
        self.dna_end_embedding = marker_sd["dna_end_embedding"].to(device=device, dtype=torch.bfloat16)

    @torch.no_grad()
    def _inject(self, input_ids, text_inputs_embeds, projected):
        """Verbatim port of BioReasonDNA.inject_dna_embeddings (marker splice + pad scatter)."""
        pad_mask = input_ids == self.dna_pad_token_id
        start_mask = input_ids == self.dna_start_token_id
        end_mask = input_ids == self.dna_end_token_id

        se = self.dna_start_embedding.view(1, 1, -1)
        ee = self.dna_end_embedding.view(1, 1, -1)
        text_inputs_embeds = torch.where(start_mask.unsqueeze(-1), se, text_inputs_embeds)
        text_inputs_embeds = torch.where(end_mask.unsqueeze(-1), ee, text_inputs_embeds)

        if pad_mask.sum() == 0:
            return text_inputs_embeds
        dna_flat = torch.cat(projected, dim=0)  # (sum pads, H)
        orig_shape = text_inputs_embeds.shape
        H = orig_shape[-1]
        embeds_2d = text_inputs_embeds.view(-1, H)
        idx = pad_mask.view(-1).nonzero(as_tuple=False).squeeze(1)
        diff = dna_flat.to(embeds_2d.dtype) - embeds_2d.index_select(0, idx)
        embeds_2d = embeds_2d.scatter_add(0, idx.unsqueeze(1).expand(-1, H), diff)
        return embeds_2d.view(orig_shape)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, dna_embeds, **kwargs):
        dna_special = ((input_ids == self.dna_start_token_id)
                       | (input_ids == self.dna_pad_token_id)
                       | (input_ids == self.dna_end_token_id))
        safe_ids = torch.where(dna_special, torch.zeros_like(input_ids), input_ids)
        text_inputs_embeds = self.text_model.get_input_embeddings()(safe_ids)
        if dna_embeds is not None:
            projected = [
                self.dna_projection(e.to(device=self.device, dtype=torch.bfloat16).unsqueeze(0)).squeeze(0)
                for e in dna_embeds
            ]
            text_inputs_embeds = self._inject(input_ids, text_inputs_embeds, projected)
        return self.text_model(inputs_embeds=text_inputs_embeds, attention_mask=attention_mask, **kwargs)


# --- Geometry (smoke) --------------------------------------------------------------------------
def _pairwise_cosine(x: torch.Tensor):
    x = x.float()
    x = x[torch.isfinite(x).all(dim=1)]
    n = x.shape[0]
    if n < 2:
        return float("nan")
    xn = torch.nn.functional.normalize(x, dim=1)
    if n > 2000:
        xn = xn[torch.randperm(n)[:2000]]
        n = 2000
    sim = xn @ xn.t()
    iu = torch.triu_indices(n, n, offset=1)
    return float(sim[iu[0], iu[1]].mean())


def _cross_cosine(a: torch.Tensor, b: torch.Tensor):
    a, b = a.float(), b.float()
    a = a[torch.isfinite(a).all(dim=1)]
    b = b[torch.isfinite(b).all(dim=1)]
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    an = torch.nn.functional.normalize(a, dim=1)
    bn = torch.nn.functional.normalize(b, dim=1)
    if an.shape[0] > 1500:
        an = an[torch.randperm(an.shape[0])[:1500]]
    if bn.shape[0] > 1500:
        bn = bn[torch.randperm(bn.shape[0])[:1500]]
    return float((an @ bn.t()).mean())


def main():  # noqa: D103
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, args.dna_stack)
    from bioreason_dna.dataset.embedding import EmbeddingStore
    from bioreason_dna.dataset.collate import dna_collate_fn
    from bioreason_dna.dataset.load import load_chat_dataset
    from bioreason_dna.dataset.format import make_sequence_id
    from sae.activation_store import ActivationStore, ActivationStoreConfig

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # --- Model ---
    print(f"[load] model from {args.ckpt_dir}")
    t0 = time.time()
    model = DNAExtractModel(args.ckpt_dir, args.dna_stack, device, args.attn_impl)
    hidden = model.text_config.hidden_size
    n_layers = model.text_config.num_hidden_layers
    print(f"[load] done in {time.time()-t0:.1f}s | layers={n_layers} hidden={hidden} "
          f"start={model.dna_start_token_id} pad={model.dna_pad_token_id} end={model.dna_end_token_id}")
    for L in args.layers:
        if not (0 <= L < n_layers):
            raise ValueError(f"layer {L} out of range [0,{n_layers})")

    # --- Data + embedding store ---
    print(f"[data] loading {args.dataset_path} (task=vep) + EmbeddingStore({args.embedding_base})")
    train_ds, val_ds, test_ds = load_chat_dataset(
        dataset_path=args.dataset_path, dataset_config=args.dataset_config, task="vep",
        truncate_dna_per_side=args.truncate_dna_per_side, num_proc=4,
    )
    splits = {"train": train_ds, "validation": val_ds, "test": test_ds}
    if args.split == "all":
        from datasets import concatenate_datasets
        chosen = [d for d in (train_ds, val_ds, test_ds) if d is not None]
        ds = concatenate_datasets(chosen)
    else:
        ds = splits[args.split]
        if ds is None:
            raise ValueError(f"split {args.split} not present")
    n = min(args.num_examples, len(ds))
    ds = ds.select(range(n))
    print(f"[data] {n} examples from split='{args.split}'")

    store = EmbeddingStore(
        base_path=args.embedding_base, roles=["reference", "variant"],
        block=args.embedding_block, prefix=args.embedding_prefix,
    )

    def collate(batch):
        return dna_collate_fn(batch, tokenizer=model.tokenizer, max_length=args.max_length_text,
                              embedding_store=store, train_mode=True)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # --- Hooks ---
    captured = {}

    def mk_hook(L):
        def hook(module, inp, out):
            captured[L] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    handles = [model.text_model.model.layers[L].register_forward_hook(mk_hook(L)) for L in args.layers]

    # --- Stores ---
    writer_pools = {L: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"wL{L}") for L in args.layers}
    stores = {L: ActivationStore(out / f"layer{L}", ActivationStoreConfig(shard_size=args.shard_size))
              for L in args.layers}
    accum = {L: ShardAccumulator(stores[L], args.shard_size, executor=writer_pools[L]) for L in args.layers}
    sidecar = LabelSidecar(out / "token_labels.parquet")
    example_meta = []

    counts = Counter()
    n_tokens_kept = 0
    n_dna_pad_dropped = 0
    fwd_time = 0.0
    global_row = 0
    rng = np.random.default_rng(args.seed)
    # smoke geometry accumulators (subsampled per-example, layers[0])
    geo_dna, geo_text, geo_cross = [], [], []
    Lg = args.layers[0]
    t_start = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="extract"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dna_embeds = batch.get("dna_embeds")  # list[Tensor] on CPU

            tf = time.time()
            model.forward(input_ids=input_ids, attention_mask=attention_mask, dna_embeds=dna_embeds)
            fwd_time += time.time() - tf

            bsz = input_ids.shape[0]
            for i in range(bsz):
                row = input_ids[i]
                pad_mask = attention_mask[i].bool()  # keep = non-pad (left-padded)
                is_dna = ((row == model.dna_start_token_id)
                          | (row == model.dna_pad_token_id)
                          | (row == model.dna_end_token_id))
                # <|dna_pad|> tokens only (excludes START/END markers, which we always keep).
                is_dna_pad = (row == model.dna_pad_token_id)
                keep = pad_mask
                keep_np = keep.cpu().numpy()
                abs_index = np.nonzero(keep_np)[0]
                token_index = abs_index - abs_index.min() if abs_index.size else abs_index
                is_dna_kept = is_dna[keep].cpu().numpy()
                bands = np.where(is_dna_kept, "dna", "text")

                # --- DNA-token subsampling: keep at most N <|dna_pad|> activations per example.
                # Text tokens and DNA START/END markers are always kept. The subsample mask
                # (over the already-kept, non-pad positions) is applied identically to the
                # activations, token_index, and bands so the sidecar stays row-aligned.
                if args.dna_subsample and args.dna_subsample > 0:
                    is_dna_pad_kept = is_dna_pad[keep].cpu().numpy()
                    dna_pad_pos = np.nonzero(is_dna_pad_kept)[0]
                    if dna_pad_pos.size > args.dna_subsample:
                        chosen = rng.choice(dna_pad_pos, size=args.dna_subsample, replace=False)
                        sub_mask = ~is_dna_pad_kept  # keep everything that is not a dna_pad
                        sub_mask[chosen] = True       # plus the sampled dna_pad positions
                        n_dna_pad_dropped += int(dna_pad_pos.size - args.dna_subsample)
                        token_index = token_index[sub_mask]
                        bands = bands[sub_mask]
                        keep_idx = np.nonzero(sub_mask)[0]
                    else:
                        keep_idx = None
                else:
                    keep_idx = None

                for L in args.layers:
                    acts = captured[L][i][keep].float().cpu().numpy()
                    if keep_idx is not None:
                        acts = acts[keep_idx]
                    accum[L].add(acts)

                sid = ds[global_row].get("sequence_id", f"row{global_row}")
                sidecar.append(str(sid), token_index, bands.tolist())
                c = Counter(bands.tolist())
                counts.update(bands.tolist())
                n_tokens_kept += int(len(bands))  # post-subsample count (rows actually stored)
                example_meta.append({
                    "sequence_id": str(sid),
                    "n_dna": int(c.get("dna", 0)),
                    "n_text": int(c.get("text", 0)),
                    "answer": str(ds[global_row].get("answer", "")),
                })

                if args.smoke:
                    h = captured[Lg][i][keep]
                    dmask_np = is_dna_kept
                    if keep_idx is not None:
                        h = h[torch.from_numpy(keep_idx).to(h.device)]
                        dmask_np = dmask_np[keep_idx]
                    dmask = torch.from_numpy(dmask_np).to(h.device)
                    dna_tok, text_tok = h[dmask], h[~dmask]
                    geo_dna.append(_pairwise_cosine(dna_tok))
                    geo_text.append(_pairwise_cosine(text_tok))
                    geo_cross.append(_cross_cosine(dna_tok, text_tok))
                global_row += 1

    for h in handles:
        h.remove()
    for L in args.layers:
        accum[L].close()
        stores[L].finalize(metadata={
            "model": "bioreason-dna-4b-vep", "layer": L, "hidden_dim": hidden,
            "n_examples": n, "split": args.split, "position_types": "dna|text",
            "hook": f"text_model.model.layers[{L}] output[0] (residual after block {L})",
        })
    for p in writer_pools.values():
        p.shutdown(wait=True)
    sidecar.close()
    pq.write_table(pa.Table.from_pylist(example_meta), str(out / "examples.parquet"))

    report = {
        "model": "bioreason-dna-4b-vep", "ckpt": args.ckpt_dir, "split": args.split,
        "layers": args.layers, "n_examples": n,
        "tokens_kept_total": n_tokens_kept, "counts_by_type": dict(counts),
        "tokens_per_example": n_tokens_kept / max(n, 1),
        "dna_subsample": args.dna_subsample,
        "dna_pad_tokens_dropped": n_dna_pad_dropped,
        "forward_time_s": round(fwd_time, 1), "wall_time_s": round(time.time() - t_start, 1),
        "hidden_dim": hidden,
    }

    def _nanmean(xs):
        v = [x for x in xs if x == x]  # drop NaN
        return float(np.mean(v)) if v else float("nan")

    if args.smoke:
        report["geometry_layer"] = Lg
        report["geometry"] = {
            "dna_self_cos": round(_nanmean(geo_dna), 4),
            "text_self_cos": round(_nanmean(geo_text), 4),
            "dna_text_cross_cos": round(_nanmean(geo_cross), 4),
            "protein_reference": {"protein_self": 0.99, "text_self": 0.54, "cross": 0.00},
        }

    (out / "extract_metadata.json").write_text(json.dumps(report, indent=2))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2))
    print("\n=== EXTRACTION REPORT ===")
    print(json.dumps(report, indent=2))
    if args.smoke:
        g = report["geometry"]
        print(f"\n[RESIDUAL STREAM @ layer {Lg}]")
        print(f"  DNA self-cos     = {g['dna_self_cos']}  (protein analog: 0.99)")
        print(f"  text self-cos    = {g['text_self_cos']}  (protein analog: 0.54)")
        print(f"  DNA<->text cross = {g['dna_text_cross_cos']}  (protein analog: 0.00)")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
