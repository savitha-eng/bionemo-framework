"""Data-only faithfulness: does the prompt's go_pred speculation already contain the ground-truth GO answer?
If yes, the model refines a given answer rather than deriving it -> explains answer-change-negative + structure-inert."""
import sys, re, numpy as np
sys.path.insert(0, "src"); import bioreason_pro_sae.data as brp
_, va, _ = brp.load_reasoning_splits(max_length_protein=2000)
GO = re.compile(r"GO:\d{7}")
recall, jac, echo = [], [], []
for i in range(len(va)):
    r = va[i]
    pred = set(GO.findall(str(r["go_pred"]))); truth = set(GO.findall(str(r["ground_truth_go_terms"])))
    ans = set(GO.findall(str(r["final_answer"])))
    if not truth: continue
    recall.append(len(pred & truth) / len(truth)); jac.append(len(pred & truth) / max(1, len(pred | truth)))
    if ans: echo.append(len(ans & pred) / len(ans))
print(f"go_pred recall of truth: {np.mean(recall):.3f} | Jaccard: {np.mean(jac):.3f} | "
      f">=80% covered: {np.mean(np.array(recall)>=0.8):.1%} | answer echo of go_pred: {np.mean(echo):.3f}")
