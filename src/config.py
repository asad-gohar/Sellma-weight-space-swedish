"""Fixed settings.

Order of decisions (see report.pdf, section 6 "Process log and deviations"):
- The prompt split and the weight-only arm (W0_LAYER, W0_SCALE, N_TOKENSET) were fixed
  before any model output was seen.
- MAX_NEW_TOKENS was raised from 64 to 96 after a 10-prompt sanity check, so that
  loops would be visible.
- The search grids live in dev_search.py / dev_search2.py. They were written after the
  dev probe, before any test-set run.
"""

SEED = 1234
MAX_NEW_TOKENS = 96          # long enough for 3-5 sentences and to expose loops
GEN_BATCH = 10

# Selection rule used on dev (dev_search.py, dev_search2.py, run_test.py): lowest
# Swedish loop rate subject to >=90% Swedish answers on sv_dev and an English-answer
# rate on en_dev >= baseline - EN_TOLERANCE. Ties: for the column edit, single-layer
# before two-layer, then the alphabetically first name; for steering and the repetition
# penalty, the alphabetically first name.
EN_TOLERANCE = 0.10

# Zero-feedback weight-only arm (W0): fixed a priori, never tuned. The scale is a
# multiple of the norm of the MEAN residual vector at that layer (50 unlabeled
# Europarl sentences); note that this is not the mean per-token norm.
W0_LAYER = 16
W0_SCALE = 0.5

N_TOKENSET = 300             # tokens per language for the unembedding direction
N_NLL_SENT = 150             # held-out Europarl sentences per language for NLL
RANDOM_SEEDS = [0, 1, 2]     # random-direction controls in run_test.py (k=3-14 are in extra_controls.py);
                             # generator seed = 1000*k + layer
