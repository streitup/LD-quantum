import torch

# Local import within package directory
from QSANNAdapter import QSANNAdapter


def test_x_only_forward():
    B, C, H, W = 2, 128, 8, 8
    num_heads = 8
    x = torch.randn(B, C, H, W)
    adapter = QSANNAdapter()
    y = adapter(x, num_heads=num_heads)
    assert isinstance(y, torch.Tensor), "adapter must return a torch.Tensor"
    assert y.shape == x.shape, f"output shape {y.shape} must equal input shape {x.shape}"
    assert y.dtype == x.dtype, f"output dtype {y.dtype} must equal input dtype {x.dtype}"
    print("[PASS] x-only forward interface: shape and dtype match")


def test_disallow_qkv():
    B, Hh, C_head, HW = 2, 8, 16, 64
    q = torch.randn(B * Hh, C_head, HW)
    k = torch.randn(B * Hh, C_head, HW)
    v = torch.randn(B * Hh, C_head, HW)
    adapter = QSANNAdapter()
    try:
        _ = adapter(q, k, v)
    except TypeError as e:
        print(f"[PASS] qkv interface disallowed as expected: {e}")
        return
    raise AssertionError("qkv interface should be disallowed in x-only mode")


if __name__ == "__main__":
    test_x_only_forward()
    test_disallow_qkv()
    print("All x-only adapter tests passed.")