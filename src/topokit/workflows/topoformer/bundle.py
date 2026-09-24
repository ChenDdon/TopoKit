"""Checksummed portable predictor: state_dict + JSON + NumPy scaler arrays."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import TopoFormerConfig


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def standardize(x, mean, scale):
    """Match StandardScaler's float32 in-place subtraction then division."""
    out = np.array(x, dtype=np.float32, copy=True)
    out -= mean
    out /= scale
    if not np.isfinite(out).all():
        raise ValueError("nonfinite standardized features")
    return out


def save_bundle(path, model, *, mean, scale, n_samples_seen, schema_path, metadata):
    import torch
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "BUNDLE.json").exists():
        raise FileExistsError(f"bundle already complete: {path}")
    dimension = int(np.prod(model.config.feature_shape))
    mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if (mean.shape != (dimension,) or scale.shape != mean.shape or
            not np.isfinite(mean).all() or not np.isfinite(scale).all() or
            np.any(scale <= 0) or int(n_samples_seen) <= 0):
        raise ValueError("invalid scaler")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path / "weights.pt")
    np.savez(path / "scaler.npz", mean=mean, scale=scale, n_samples_seen=int(n_samples_seen))
    (path / "config.json").write_text(json.dumps(model.config.to_dict(), indent=2) + "\n")
    (path / "feature_schema.json").write_bytes(Path(schema_path).read_bytes())
    files = {name: sha256(path / name) for name in
             ("weights.pt", "scaler.npz", "config.json", "feature_schema.json")}
    receipt = {"format": "topokit.topoformer.bundle.v1", "implementation": "compact-supervised-v1",
               "files": files, "metadata": metadata, "torch_version": str(torch.__version__)}
    tmp = path / "BUNDLE.json.tmp"
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path / "BUNDLE.json")
    return receipt


class Predictor:
    def __init__(self, path, *, device="cpu"):
        import torch
        from . import build_model
        path = Path(path)
        self.receipt = json.loads((path / "BUNDLE.json").read_text())
        if self.receipt.get("format") != "topokit.topoformer.bundle.v1":
            raise ValueError("unsupported bundle")
        expected = {"weights.pt", "scaler.npz", "config.json", "feature_schema.json"}
        if set(self.receipt["files"]) != expected:
            raise ValueError("incomplete bundle file inventory")
        for name, digest in self.receipt["files"].items():
            if sha256(path / name) != digest:
                raise ValueError(f"bundle checksum mismatch: {name}")
        self.config = TopoFormerConfig(**json.loads((path / "config.json").read_text()))
        with np.load(path / "scaler.npz", allow_pickle=False) as state:
            self.mean = state["mean"].copy()
            self.scale = state["scale"].copy()
            self.n_samples_seen = int(state["n_samples_seen"])
        d = int(np.prod(self.config.feature_shape))
        if (self.mean.shape != (d,) or self.scale.shape != (d,) or
                not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or
                np.any(self.scale <= 0) or self.n_samples_seen <= 0):
            raise ValueError("invalid scaler in bundle")
        self.device = torch.device(device)
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu", weights_only=True), strict=True)
        self.model.eval()

    def predict(self, features, *, schema_sha256, batch_size=32):
        """Predict raw feature tensors; require their producer's schema identity."""
        import torch
        if schema_sha256 != self.receipt["files"]["feature_schema.json"]:
            raise ValueError("feature schema does not match model")
        x = np.asarray(features)
        if x.ndim != 4 or tuple(x.shape[1:]) != self.config.feature_shape or batch_size <= 0:
            raise ValueError("invalid input shape or batch size")
        result = []
        with torch.inference_mode():
            for begin in range(0, len(x), batch_size):
                raw = x[begin:begin + batch_size]
                normalized = standardize(raw.reshape(len(raw), -1), self.mean, self.scale)
                tensor = torch.from_numpy(normalized.reshape(raw.shape)).to(self.device)
                result.append(self.model(tensor).cpu().numpy())
        out = np.concatenate(result) if result else np.empty(0, dtype=np.float32)
        if not np.isfinite(out).all():
            raise ValueError("nonfinite predictions")
        return out
