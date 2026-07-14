from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dynanchor.io_formats import read_fvecs, write_fvecs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export hub embeddings from a saved official GATE artifact model.")
    parser.add_argument("--gate-root", type=Path, default=ROOT / "third_party" / "GATE" / "jacksondca-gate-78334b4")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ep-vectors", type=Path, required=True)
    parser.add_argument("--graph-features-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.gate_root.resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model, map_location=device, weights_only=False).to(device)
    model.eval()
    ep_vectors = torch.tensor(read_fvecs(args.ep_vectors), dtype=torch.float32, device=device)
    graph_features = torch.tensor(pd.read_csv(args.graph_features_csv, index_col=0).to_numpy(dtype="float32"), dtype=torch.float32, device=device)
    with torch.no_grad():
        embeddings = model(ep_vectors, graph_features).detach().cpu().numpy().astype("float32")
    write_fvecs(args.out, embeddings)
    print({"model": str(args.model), "out": str(args.out), "rows": int(embeddings.shape[0]), "dim": int(embeddings.shape[1]), "device": str(device)})


if __name__ == "__main__":
    main()
