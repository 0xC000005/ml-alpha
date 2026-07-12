---
id: L-05
status: accepted
supersedes: null
---

# L-05 — Attention is not the memory bottleneck at this scale

- **L-05 (attention scaling, workflow wqq713yie):** The cross-section is a permutation-invariant SET (no positional encoding anywhere) and at N≈5000 it is **NOT attention-bound** (block fwd+bwd ≈1.4 GB; the 16–25 GB is gradients/AMP/optimizer/rolling-window data). So **KV-cache** (an autoregressive-decoding optimization — this is a single-pass non-causal encoder, nothing to cache) and **sequence-sparse** patterns (Longformer/BigBird sliding-window/strided — assume an ordering the set lacks; would mask arbitrary stocks and destroy signal-bearing cross-stock edges) are **misapplied here**. The long-context curse was tamed by **exact FlashAttention**, not approximation. Free exact win TAKEN: `need_weights=False` at the MHA calls → drops the discarded (1,N,N) tensor + dispatches the fused SDPA/FlashAttention kernel (applied in `experiments/exp_transformer.py`; outputs match to ~2.6e-7). The real O(N²) cliff is the **TEMPORAL extension**: naive tokens = stock×month → (N·T)² ≈ 29 GB fp16 scores at T=24 (~576×). **Design it out** — per-stock temporal encoder (GRU/TCN) collapsing each stock's trailing window to ONE token so token count stays N (cheapest; macro-GRU B-02 is the wired first step), or exact axial/factorized attention; never build the flat (N·T)² map then approximate. ISAB inducing points (B-03) only matter past ~10–15k stocks/month; Linformer is invalid (its fixed projection breaks when N varies monthly). Approximate attention must beat exact on the OOS seed×year distribution before adoption (L-01).
