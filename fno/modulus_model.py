"""Datacenter-cooling FNO on NVIDIA PhysicsNeMo (formerly Modulus).
Same operator and same config as our from-scratch fno2d_cool.py, but built
with the API the Modulus/PhysicsNeMo team ships and maintains."""
from physicsnemo.models.fno import FNO

T_IN, N_FIELDS, N_STATIC = 10, 2, 1          # match the from-scratch model exactly
IN_CH = T_IN * N_FIELDS + N_STATIC           # 21: [w,T] x 10 frames, + source map S
OUT_CH = N_FIELDS                            # 2 : [w, T] at t+1


def build_model(modes=12, width=20):
    return FNO(
        in_channels=IN_CH,
        out_channels=OUT_CH,
        dimension=2,
        latent_channels=width,        # == our `width`
        num_fno_layers=4,             # == our 4 Fourier layers
        num_fno_modes=modes,          # == our `modes`
        padding=0,                    # our data is periodic; no domain padding needed
        decoder_layers=1,
        decoder_layer_size=width * 2,
        coord_features=True,          # Modulus appends grid coords for you
    )
