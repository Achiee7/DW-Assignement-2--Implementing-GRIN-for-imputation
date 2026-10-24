"""Execute the submission notebook in the current Python environment."""
from argparse import ArgumentParser
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['bounded', 'paper'], default='bounded')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--retrain', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    original = root / 'notebooks' / 'grin_aqi36_reproduction.ipynb'
    book = nbformat.read(original, as_version=4)
    setup = next(cell for cell in book.cells if cell.cell_type == 'code' and 'PROFILE =' in cell.source)
    setup.source = setup.source.replace("PROFILE = 'bounded'", f"PROFILE = {args.profile!r}")
    setup.source = setup.source.replace('seed=42', f'seed={args.seed}')
    setup.source = setup.source.replace('RETRAIN = False', f'RETRAIN = {args.retrain!r}')
    setup.source = setup.source.replace("OUTPUT = ROOT", f"CONFIG.device = {args.device!r}\nOUTPUT = ROOT", 1)
    if args.profile == 'bounded' and args.seed == 42 and args.device == 'cpu' and not args.retrain:
        destination = original
    else:
        destination = root / 'notebooks' / 'runs' / f'grin_{args.profile}_{args.seed}_{args.device}.ipynb'
        destination.parent.mkdir(parents=True, exist_ok=True)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(book, timeout=172800, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}})
    client.km = manager
    try:
        client.execute()
    finally:
        nbformat.write(book, destination)
    print(f'Executed notebook saved to {destination}')


if __name__ == '__main__':
    main()
