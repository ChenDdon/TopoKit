"""Representation guards for the optional sequence modality."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from topokit.workflows.protein_ligand_prediction.sequence import (
    concatenate_embeddings, ligand_tokens, protein_windows, weighted_mean,
)

VOCAB={x:i for i,x in enumerate(['<s>','<pad>','</s>','<unk>','C','O','F','[',']','+','2'])}


def test_unknown_element_is_one_unk_not_partial_fluorine():
    with pytest.raises(ValueError,match='Unsupported'):ligand_tokens('[Fe+2]',VOCAB)
    ids,receipt=ligand_tokens('[Fe+2]',VOCAB,allow_unknown=True)
    assert ids==[0,7,3,9,10,8,2]
    assert receipt['unknown_symbols']==['Fe'] and receipt['unknown_count_encoded']==1


def test_truncation_requires_opt_in_and_preserves_eos():
    with pytest.raises(ValueError,match='limit'):ligand_tokens('C'*255,VOCAB)
    ids,receipt=ligand_tokens('C'*255,VOCAB,allow_truncation=True)
    assert len(ids)==256 and ids[-1]==2 and receipt['tokens_omitted']==1
    assert receipt['tokens_total']==255


@pytest.mark.parametrize('bad',['',' C','C?C','C C'])
def test_no_silent_lexical_loss(bad):
    with pytest.raises(ValueError):ligand_tokens(bad,VOCAB,allow_unknown=True)


def test_windows_keep_all_residues_and_weight_correctly():
    seq='A'*1022+'CG'; assert protein_windows(seq)==['A'*1022,'CG']
    pooled=weighted_mean([[2.,4.],[8.,10.]],[1022,2])
    np.testing.assert_allclose(pooled,[(2044+16)/1024,(4088+20)/1024])
    assert pooled.dtype==np.float32
    # Duplicated biological chains retain their multiplicity during pooling.
    np.testing.assert_allclose(weighted_mean([[1],[3],[1]],[2,2,2]),[5/3])


def test_feature_order_and_finite_checks():
    v=concatenate_embeddings(np.ones(1280),np.full(512,2));assert v.shape==(1792,)
    assert v.dtype==np.float32 and np.all(v[:1280]==1) and np.all(v[1280:]==2)
    with pytest.raises(ValueError):concatenate_embeddings(np.full(1280,np.nan),np.zeros(512))
    with pytest.raises(ValueError):weighted_mean([[1],[2]],[0,1])


def test_optional_libraries_remain_lazy():
    script='import sys; import topokit.workflows.protein_ligand_prediction.sequence; assert not any(x in sys.modules for x in ["torch","transformers","rdkit","sklearn"])'
    subprocess.run([sys.executable,'-c',script],check=True)


def test_training_scaler_uses_fit_rows_only():
    pytest.importorskip('sklearn')
    from topokit.workflows.protein_ligand_prediction.sequence import make_gbdt
    x=np.arange(120,dtype=np.float32).reshape(20,6);model=make_gbdt(n_estimators=2).fit(x,np.arange(20))
    mean=model[0].mean_.copy();model.predict(np.full((3,6),1e8))
    np.testing.assert_array_equal(model[0].mean_,mean)
    np.testing.assert_allclose(mean,x.mean(0))
    assert model[1].n_iter_no_change is None


def test_original_supported_smiles_tokens_and_checkpoint_bos():
    torch=pytest.importorskip('torch')
    from topokit.workflows.protein_ligand_prediction.sequence import SequenceEncoder
    from topokit.workflows.protein_ligand_prediction._chembl import tokenize_smiles
    root=Path(__file__).resolve().parents[2]/'datasets/protein_ligand_prediction/pretrained/sequence/PretrainModels'
    if not (root/'chembl27_512/checkpoint_best.pt').exists():pytest.skip('External checkpoint not installed')
    torch.set_num_threads(2)
    encoder=SequenceEncoder(chembl_dir=root/'chembl27_512').load_ligand()
    smiles=(root/'outputs/validation_chembl27_512/smoke.smi').read_text().splitlines()
    for s in smiles:
        ids,_=ligand_tokens(s,encoder.vocab.indices)
        assert ids==[0]+[encoder.vocab.index(t) for t in tokenize_smiles(s)]+[2]
    result,_=encoder.ligand_batch(smiles)
    expected=np.load(root/'outputs/validation_chembl27_512/smoke_bos.npy')
    np.testing.assert_allclose(result,expected,atol=3e-5,rtol=3e-5)
    single,_=encoder.ligand_batch([smiles[0]])
    np.testing.assert_allclose(single[0],result[0],atol=3e-5,rtol=3e-5)
