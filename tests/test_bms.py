import numpy as np
from persystems.gm import GenerativeModel
from persystems.bms import log_evidence_discrete, compare_models_ffx, compare_models_rfx

def test_bms_prefers_true_model():
    # Build two models differing in A (likelihood); generate data from model 0
    N = 5
    gm0 = GenerativeModel.make_ring_world(N=N, A_eps=0.10, target_idx=3)
    gm1 = GenerativeModel.make_ring_world(N=N, A_eps=0.30, target_idx=3)
    models = [{"A": gm0.A, "B": gm0.B, "name": "Aeps0.10"},
              {"A": gm1.A, "B": gm1.B, "name": "Aeps0.30"}]

    # Generate a short run of actions+observations under gm0
    rng = np.random.default_rng(0)
    T = 40
    qs = np.ones(N)/N
    true_s = rng.integers(0, N)
    acts = []
    obs  = []
    for t in range(T):
        a = rng.integers(0, len(gm0.B))   # random actions
        true_s = (true_s + gm0.actions[a]) % N
        o = int(rng.choice(np.arange(N), p=gm0.A[:, true_s]))
        acts.append(a)
        obs.append(o)

    le0 = log_evidence_discrete(gm0.A, gm0.B, acts, obs)
    le1 = log_evidence_discrete(gm1.A, gm1.B, acts, obs)
    assert le0 > le1  # true model should win on average

    # FFX
    out = compare_models_ffx(models, [(acts, obs)])
    assert out["post"].argmax() == 0

    # RFX
    out_r = compare_models_rfx(models, [(acts, obs)], alpha0=1.0, nsamples=2000)
    assert out_r["exceedance"][0] > out_r["exceedance"][1]

