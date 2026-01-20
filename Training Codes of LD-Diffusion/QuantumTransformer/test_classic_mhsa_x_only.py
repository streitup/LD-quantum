import torch
from ClassicMHSAAdapter import ClassicMHSAAdapter


def test_forward_shapes():
    B, C, H, W = 2, 128, 8, 8
    num_heads = 8
    x = torch.randn(B, C, H, W)
    adapter = ClassicMHSAAdapter(attn_dropout=0.1)
    y = adapter(x, num_heads=num_heads)
    assert isinstance(y, torch.Tensor)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_amp_compat():
    if not torch.cuda.is_available():
        return
    B, C, H, W = 2, 128, 8, 8
    num_heads = 8
    x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float16)
    adapter = ClassicMHSAAdapter(attn_dropout=0.0, proj_dropout=0.0)
    y = adapter(x, num_heads=num_heads)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


if __name__ == '__main__':
    test_forward_shapes()
    test_amp_compat()
    print('ClassicMHSAAdapter x-only tests passed.')