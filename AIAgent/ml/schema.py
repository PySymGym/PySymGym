"""The graph tensor schema: feature widths shared by the dataset and models.

The dataset layer (``ml.dataset``) produces the per-vertex-type feature
matrices and the models consume them; both sides must agree on each matrix's
width. These constants are the single source of truth for those widths —
neither the producer nor a model may hard-code them.
"""

NUM_PC_FEATURES = 48
NUM_STATE_FEATURES = 6
NUM_CFG_VERTEX_FEATURES = 7
