"""Load the authors' unchanged model from a verified, pinned external checkout.

No upstream source is redistributed in this submission. Only the modules actually
needed by GRIN are downloaded; legacy Lightning/PyG package initializers are bypassed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
from urllib.request import urlopen

COMMIT = "4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e"
REPOSITORY = "Graph-Machine-Learning-Group/grin"
MANIFEST = json.loads(Path(__file__).with_name("upstream_manifest.json").read_text())


def fetch_sources(cache_dir: str | Path | None = None) -> Path:
    """Fetch and hash-check immutable source files, including existing cache entries."""
    root = Path(cache_dir or Path.home() / ".cache" / "grin-reproduction" / COMMIT)
    for relative, expected in MANIFEST.items():
        path = root / relative
        if not path.exists():
            payload = urlopen(f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{relative}", timeout=60).read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise RuntimeError(f"Downloaded source hash mismatch: {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Cached source hash mismatch: {path}")
    return root


def load_grin_class(cache_dir: str | Path | None = None):
    """Import unchanged GRINet in a private namespace without executing lib/__init__."""
    root = fetch_sources(cache_dir)
    namespace = "_verified_grin_" + COMMIT[:12]
    for suffix in ("", ".nn", ".nn.layers", ".nn.models", ".nn.utils", ".utils"):
        name = namespace + suffix
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root / "lib" / suffix.strip(".").replace(".", "/"))]
            sys.modules[name] = package
    sys.modules[namespace].epsilon = 1e-8
    for relative in MANIFEST:
        name = namespace + "." + relative.removeprefix("lib/").removesuffix(".py").replace("/", ".")
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, root / relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        if relative.endswith("layers/gril.py"):
            sys.modules[namespace + ".nn.layers"].BiGRIL = sys.modules[name].BiGRIL
    return sys.modules[namespace + ".nn.models.grin"].GRINet


def make_model(adjacency, hidden_size=64, embedding_size=8, cache_dir=None):
    """The AQI-36 architecture from config/grin/air36.yaml."""
    return load_grin_class(cache_dir)(
        adj=adjacency, d_in=1, d_hidden=hidden_size, d_ff=64, ff_dropout=0.,
        n_layers=1, kernel_size=2, decoder_order=1, global_att=False,
        d_emb=embedding_size, layer_norm=False, merge="mlp", impute_only_holes=True)
