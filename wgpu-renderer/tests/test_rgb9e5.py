import numpy as np

from replica_wgpu.scene import load_hdr_atlas_rgb9e5


def decode(packed):
    m = np.stack([(packed >> s) & 0x1FF for s in (0, 9, 18)], axis=-1).astype(np.float64)
    e = (packed >> 27).astype(np.float64)
    return m * np.exp2(e - 15 - 9)[..., None]


def test_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    dim = 64
    rgb = (rng.random((dim, dim, 3)) ** 4 * 2000).astype(np.float16)  # HDR-ish spread
    rgb[0, 0] = 0
    rgb[0, 1] = [65504, 1, 0]  # beyond the RGB9E5 range: clamps
    path = tmp_path / "0-color-ptex.hdr"
    rgb.tofile(path)

    packed, d = load_hdr_atlas_rgb9e5(str(path))
    assert d == dim
    out = decode(packed)
    ref = np.clip(rgb.astype(np.float64), 0, 65408)
    np.testing.assert_array_equal(out[0, 0], 0)
    # Each channel is within half a step of the shared exponent's quantum.
    quantum = np.exp2((packed >> 27).astype(np.float64) - 15 - 9)[..., None]
    assert np.all(np.abs(out - ref) <= 0.5 * quantum + 1e-12)
