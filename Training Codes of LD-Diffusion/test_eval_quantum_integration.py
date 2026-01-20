import argparse
import torch
from training.networks_eval import UNetBlock
from QuantumTransformer.QSANNAdapter import QSANNAdapter


def run_block_forward(use_quantum: bool, num_heads: int = 1,
                      N_QUBITS: int = 8, Q_DEPTH: int = 4, encoding: str = 'amplitude'):
    B, C, H, W = 2, 64, 32, 32
    emb_channels = 128

    # Prepare inputs
    x = torch.randn(B, C, H, W)
    emb = torch.randn(B, emb_channels)

    adapter = None
    if use_quantum:
        adapter = QSANNAdapter(N_QUBITS=N_QUBITS, Q_DEPTH=Q_DEPTH, encoding=encoding)

    # Instantiate block
    block = UNetBlock(
        in_channels=C,
        out_channels=C,
        emb_channels=emb_channels,
        attention=True,
        num_heads=num_heads,
        use_quantum_transformer=use_quantum,
        quantum_adapter=adapter,
    )

    y = block(x, emb)
    assert isinstance(y, torch.Tensor), 'Output must be a tensor'
    assert y.shape == x.shape, f'Output shape mismatch: {y.shape} vs {x.shape}'
    assert y.dtype == x.dtype, f'Output dtype mismatch: {y.dtype} vs {x.dtype}'
    print(f"Forward ok | quantum={use_quantum} | shape={y.shape} | dtype={y.dtype}")


def main():
    parser = argparse.ArgumentParser(description='Integration test for UNetBlock with QSANNAdapter (evaluation path).')
    parser.add_argument('--use-quantum', action='store_true', help='Run quantum path (default runs both).')
    parser.add_argument('--num-heads', type=int, default=1, help='Number of attention heads (default: 1).')
    parser.add_argument('--n-qubits', type=int, default=8, help='QSANNAdapter N_QUBITS (default: 8).')
    parser.add_argument('--q-depth', type=int, default=4, help='QSANNAdapter Q_DEPTH (default: 4).')
    parser.add_argument('--encoding', type=str, default='amplitude', choices=['amplitude','angle'], help='QSANNAdapter encoding (default: amplitude).')
    args = parser.parse_args()

    if args.use_quantum:
        run_block_forward(use_quantum=True, num_heads=args.num_heads, N_QUBITS=args.n_qubits, Q_DEPTH=args.q_depth, encoding=args.encoding)
    else:
        # Run both classical and quantum paths with defaults
        run_block_forward(use_quantum=False, num_heads=args.num_heads)
        run_block_forward(use_quantum=True, num_heads=args.num_heads, N_QUBITS=args.n_qubits, Q_DEPTH=args.q_depth, encoding=args.encoding)


if __name__ == '__main__':
    main()