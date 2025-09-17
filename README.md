# persystems — Minimal Active Inference (Phase 1)

This phase provides a clean ring-world demo of active inference with:
- Discrete generative model {A, B, C}
- Bayes filtering for state beliefs
- Action selection by Expected Free Energy (EFE)
  - Two decompositions: **risk+ambiguity** and **cost − information gain**
- Optional depth-H planning (exact, for small problems)
- Tests to sanity-check stochasticity and decomposition identity

## Install
```bash
pip install -r requirements.txt

