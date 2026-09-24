"""Frozen sequence encoders for a separate protein–ligand GBDT modality.

Weights are companion data assets, never downloaded during import/inference.
PyTorch/Transformers are optional and imported only when constructing an encoder.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

FEATURE_SIZE = 1792
CHEMBL_SHA256 = "8f4b94db0b35514068b755313a1630233041a1f99714003ad76c530dbc979924"
CHEMBL_DICT_SHA256 = "65fd0c5ca8af7403bd170155c9740b63273fb203ef347c856a0e9d72045acc7c"
CPZ_SHA256 = "c47887518a1c72591445f972b57145f68c820bbe50f7f3c36ed8d4e56b7add1d"
CPZ_DICT_SHA256 = "beccc7cd81c5e0e921bd55b918eaca34b54433aa969cac77b62b7817c04f4dcd"
ESM_SHA256 = "a08adabb949fa67ad3c14b509d04fd60368b35007b0095e3358f81200c4f4db0"
ESM_METADATA = {
    'config.json':'539095c22efc52a09d6147074ba4ca119f76a890df5901213b2b55f7d2f96b2b',
    'special_tokens_map.json':'3aedcd4211c0d43aec4e607ff60a63255f3174ead795e997350f09a5f8cd9ee1',
    'tokenizer_config.json':'7e9161ecdb548ec45a41cbc6b24aa4476fdd418461f491c4207baa99419a29ad',
    'vocab.txt':'0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03',
}
RECIPE_ID = "sequence-esm2-t33-chembl27-bos-v1"
CPZ_RECIPE_ID = "sequence-esm2-t33-cpz-bos-v1"
_LIGAND_PROFILES = {
    "chembl27": ("ChEMBL27", CHEMBL_SHA256, CHEMBL_DICT_SHA256, RECIPE_ID),
    "cpz": ("CPZ", CPZ_SHA256, CPZ_DICT_SHA256, CPZ_RECIPE_ID),
}
# Full lexical coverage, including unsupported element symbols as whole tokens.
# Supported SMILES retain the original ChEMBL token boundaries (% and digits
# are separate tokens). Unknown symbols map to the checkpoint's existing UNK.
_ELEMENTS = 'He Li Be Ne Na Mg Al Si Cl Ar Ca Sc Ti Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split()
_INSIDE = re.compile('|'.join(_ELEMENTS) + r"|se|as|te|[A-Zbcnops0-9@+\-:*]")
_OUTSIDE = re.compile(r"Cl|Br|[A-Zbcnops0-9#%)(+\-\\/.=@\]:*~<>$]")


def _lex(smiles):
    tokens=[]; pos=0; inside=False
    while pos < len(smiles):
        char=smiles[pos]
        if char=='[' and not inside:
            inside=True;tokens.append(char);pos+=1;continue
        if char==']' and inside:
            inside=False;tokens.append(char);pos+=1;continue
        match=(_INSIDE if inside else _OUTSIDE).match(smiles,pos)
        if match is None:raise ValueError('SMILES contains characters outside the audited lexer')
        tokens.append(match.group());pos=match.end()
    if inside:raise ValueError('Unclosed SMILES atom bracket')
    return tokens


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ligand_tokens(smiles, indices, *, max_positions=256, allow_truncation=False,
                  allow_unknown=False):
    """Return token IDs and an explicit representation-loss receipt.

    Production must opt into upstream-compatible first-254-token truncation
    and UNK handling. Defaults reject either case. No characters are discarded
    by the lexer. Canonicalization belongs to the input preparation stage.
    """
    if not isinstance(smiles, str) or not smiles or smiles != smiles.strip():
        raise ValueError("SMILES must be a nonempty stripped string")
    tokens = _lex(smiles)
    if "".join(tokens) != smiles:
        raise ValueError("SMILES contains characters outside the audited lexer")
    unknown = sorted(set(tokens) - set(indices))
    if unknown and not allow_unknown:
        raise ValueError(f"Unsupported ligand tokens: {unknown}")
    if max_positions < 3:
        raise ValueError("max_positions must leave room for BOS, content, EOS")
    truncated = max(0, len(tokens) - (max_positions - 2))
    if truncated and not allow_truncation:
        raise ValueError(f"Ligand exceeds model limit by {truncated} tokens")
    ids = [indices["<s>"]]
    ids.extend(indices.get(t, indices["<unk>"]) for t in tokens[:max_positions - 2])
    ids.append(indices["</s>"])
    return ids, {"tokens_total": len(tokens), "tokens_omitted": truncated,
                 "unknown_symbols": unknown,
                 "unknown_count_encoded": sum(t not in indices for t in tokens[:max_positions - 2])}


def protein_windows(sequence, *, window_size=1022):
    """Full-coverage, nonoverlapping residue windows; no sequence truncation."""
    if not isinstance(sequence, str) or not sequence or not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYXBZUO]+", sequence):
        raise ValueError("Expected a nonempty uppercase amino-acid sequence")
    if not isinstance(window_size, int) or not 1 <= window_size <= 1022:
        raise ValueError("window_size must be an integer in [1, 1022]")
    return [sequence[i:i + window_size] for i in range(0, len(sequence), window_size)]


def weighted_mean(vectors, lengths):
    """Pool means with residue counts, retaining repeated biological chains."""
    x = np.asarray(vectors, dtype=np.float64)
    w = np.asarray(lengths, dtype=np.float64)
    if x.ndim != 2 or w.shape != (len(x),) or not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Finite embedding rows and positive matching lengths required")
    return np.average(x, axis=0, weights=w).astype(np.float32)


def concatenate_embeddings(protein, ligand):
    p, l = np.asarray(protein), np.asarray(ligand)
    if p.shape != (1280,) or l.shape != (512,) or not np.isfinite(p).all() or not np.isfinite(l).all():
        raise ValueError("Expected finite protein[1280] and ligand[512] embeddings")
    return np.concatenate((p, l)).astype(np.float32)


class SequenceEncoder:
    """ESM-2 residue mean + a verified ChEMBL27 or CPZ ligand BOS vector.

    Load one encoder at a time with ``load_protein``/``load_ligand`` to reduce
    memory. ``encode`` is the convenient single-complex route. Full-benchmark
    runners cache identical sequences/SMILES and use the batch methods.

    ``ligand_profile="chembl27"`` preserves the historical FS-AQ default.
    Select ``ligand_profile="cpz"`` explicitly for the ESM-2/CPZ features used
    by FS-AU. ``chembl_dir`` holds the corresponding checkpoint and dictionary
    for either profile. ``recipe_id`` identifies features, not a fitted model;
    this selection does not change ``make_gbdt`` or ``predict_gbdt`` defaults.
    """
    def __init__(self, *, esm_dir=None, chembl_dir=None, device="cpu",
                 ligand_profile="chembl27"):
        if ligand_profile not in _LIGAND_PROFILES:
            raise ValueError("ligand_profile must be 'chembl27' or 'cpz'")
        self.esm_dir = Path(esm_dir) if esm_dir else None
        self.chembl_dir = Path(chembl_dir) if chembl_dir else None
        self.device = device
        self._ligand_profile = ligand_profile
        self.protein_model = self.ligand_model = None

    @property
    def ligand_profile(self):
        return self._ligand_profile

    @property
    def recipe_id(self):
        return _LIGAND_PROFILES[self._ligand_profile][3]

    def load_protein(self):
        import torch
        from transformers import AutoTokenizer, EsmModel
        if self.esm_dir is None or sha256_file(self.esm_dir / "model.safetensors") != ESM_SHA256:
            raise ValueError("ESM-2 checkpoint identity mismatch")
        for name,digest in ESM_METADATA.items():
            if sha256_file(self.esm_dir / name)!=digest:
                raise ValueError(f'ESM tokenizer/config identity mismatch: {name}')
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.esm_dir), local_files_only=True, trust_remote_code=False)
        self.protein_model, info = EsmModel.from_pretrained(
            str(self.esm_dir), local_files_only=True, add_pooling_layer=False,
            output_loading_info=True, trust_remote_code=False)
        if info["missing_keys"] or info.get("mismatched_keys") or any(not key.startswith("lm_head.") for key in info["unexpected_keys"]):
            raise ValueError(f"Incomplete ESM loading: {info}")
        if self.protein_model.config.hidden_size != 1280 or self.protein_model.config.num_hidden_layers != 33:
            raise ValueError("Incorrect ESM architecture")
        self.protein_model.to(torch.device(self.device)).eval().requires_grad_(False)
        return self

    def load_ligand(self):
        label, checkpoint_sha, vocabulary_sha, _ = _LIGAND_PROFILES[self._ligand_profile]
        if self.chembl_dir is None or sha256_file(self.chembl_dir / "checkpoint_best.pt") != checkpoint_sha:
            raise ValueError(f"{label} checkpoint identity mismatch")
        if sha256_file(self.chembl_dir / "dict.txt") != vocabulary_sha:
            raise ValueError(f"{label} vocabulary identity mismatch")
        import torch
        from ._chembl import build_model_from_checkpoint
        self.ligand_model, self.vocab, self.ligand_config = build_model_from_checkpoint(
            str(self.chembl_dir / "checkpoint_best.pt"), str(self.chembl_dir / "dict.txt"), torch.device(self.device))
        if self.ligand_config.embed_dim != 512 or self.ligand_config.max_positions != 256:
            raise ValueError(f"Incorrect {label} architecture")
        self.ligand_model.requires_grad_(False)
        return self

    def protein_batch(self, windows):
        import torch
        if not windows or any(len(protein_windows(w)) != 1 for w in windows):
            raise ValueError("Supply nonempty windows of at most 1022 residues")
        if self.protein_model is None:
            self.load_protein()
        encoded = self.tokenizer(list(windows), padding=True, return_tensors="pt", return_special_tokens_mask=True)
        residue_mask = encoded.pop("special_tokens_mask").eq(0) & encoded["attention_mask"].bool()
        if residue_mask.sum(dim=1).tolist() != [len(w) for w in windows]:
            raise ValueError("ESM tokenizer changed residue counts")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.inference_mode():
            hidden = self.protein_model(**encoded).last_hidden_state
            mask = residue_mask.to(self.device).unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        result = pooled.cpu().float().numpy()
        if result.shape != (len(windows), 1280) or not np.isfinite(result).all():
            raise ValueError("Invalid ESM output")
        return result

    def ligand_batch(self, smiles, *, allow_truncation=False, allow_unknown=False):
        import torch
        if not smiles:
            raise ValueError("Nonempty SMILES batch required")
        if self.ligand_model is None:
            self.load_ligand()
        rows = [ligand_tokens(s, self.vocab.indices, allow_truncation=allow_truncation,
                              allow_unknown=allow_unknown) for s in smiles]
        tokens = torch.full((len(rows), max(len(ids) for ids, _ in rows)), self.vocab.pad_index,
                            dtype=torch.long, device=self.device)
        for i, (ids, _) in enumerate(rows):
            tokens[i, :len(ids)] = torch.tensor(ids, device=self.device)
        with torch.inference_mode():
            result = self.ligand_model(tokens)[:, 0, :].cpu().float().numpy()
        if result.shape != (len(smiles), 512) or not np.isfinite(result).all():
            raise ValueError("Invalid ChEMBL output")
        return result, [receipt for _, receipt in rows]

    def encode(self, protein_chains, smiles, *, allow_truncation=False, allow_unknown=False):
        windows = [w for chain in protein_chains for w in protein_windows(chain)]
        # Single-window inference keeps the convenience route's memory bounded.
        protein = weighted_mean([self.protein_batch([w])[0] for w in windows], [len(w) for w in windows])
        ligand, receipts = self.ligand_batch([smiles], allow_truncation=allow_truncation, allow_unknown=allow_unknown)
        return concatenate_embeddings(protein, ligand[0]), receipts[0]


def make_gbdt(*, n_estimators=10000):
    """The existing benchmark settings; scaler is fitted only inside fit()."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), GradientBoostingRegressor(
        n_estimators=n_estimators, learning_rate=0.002, max_depth=7,
        max_features="sqrt", min_samples_split=5, subsample=0.8,
        random_state=0, n_iter_no_change=None))


def fit_gbdt(features, labels):
    """Fit all supplied training rows with the fixed 10,000-tree settings.

    Supply only the training membership; this function performs no split,
    validation tuning, target scaling, or feature imputation.
    """
    x=np.asarray(features,dtype=np.float32);y=np.asarray(labels,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=FEATURE_SIZE or y.shape!=(len(x),) or len(x)<5 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite training features[n,1792] and matching labels required')
    return make_gbdt().fit(x,y)


def predict_gbdt(features, model_dir):
    """Predict with a trusted sequence-study bundle and its embedded scaler.

    Loading joblib requires a trusted publisher; a checksum verifies integrity,
    not the trustworthiness of an arbitrary external pickle.
    """
    import json
    import joblib
    import sklearn
    folder=Path(model_dir);receipt=json.loads((folder/'RECEIPT.json').read_text())
    if receipt.get('recipe')!=RECIPE_ID or receipt.get('status')!='passed' or receipt.get('feature_size')!=FEATURE_SIZE:
        raise ValueError('Bundle does not contain this sequence strategy')
    if receipt['sklearn']!=sklearn.__version__:
        raise ValueError(f"Model requires scikit-learn {receipt['sklearn']}; installed {sklearn.__version__}")
    if sha256_file(folder/'pipeline.joblib')!=receipt['pipeline_sha256']:raise ValueError('Model checksum mismatch')
    x=np.asarray(features,dtype=np.float32)
    if x.ndim==1:x=x[None,:]
    if x.ndim!=2 or x.shape[1]!=FEATURE_SIZE or not np.isfinite(x).all():raise ValueError('Expected finite features[n,1792]')
    model=joblib.load(folder/'pipeline.joblib')
    if model[0].n_features_in_!=FEATURE_SIZE or model[1].n_estimators_!=10000:raise ValueError('Model dimensions/tree count mismatch')
    return model.predict(x)
