"""Independent layout, optional-dependency, model and portable-bundle checks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from topokit.workflows.topoformer import TopoFormerConfig, position_encoding, tokenize


def test_scale_tokens_preserve_statistic_channel_order():
    features = np.arange(2*3*5*7).reshape(2,3,5,7)
    tokens = tokenize(features)
    for b in range(2):
        for s in range(5):
            np.testing.assert_array_equal(tokens[b,s], [features[b,t,s,c] for t in range(3) for c in range(7)])
    assert not np.array_equal(tokens, features.reshape(2,5,21))


@pytest.mark.parametrize('width',[128,256,512])
def test_positions_match_width_first_meshgrid_oracle(width):
    grid_w, grid_h = np.meshgrid(np.arange(1,dtype=np.float32),np.arange(50,dtype=np.float32))
    axes=[]
    for axis in (grid_w,grid_h):
        freq=np.power(10000.,-np.arange(width//4,dtype=float)/(width//4))
        phase=np.einsum('m,d->md',axis.reshape(-1),freq)
        axes.append(np.concatenate([np.sin(phase),np.cos(phase)],axis=1))
    oracle=np.vstack([np.zeros((1,width)),np.hstack(axes)]).astype(np.float32)
    np.testing.assert_allclose(position_encoding(50,width),oracle,rtol=0,atol=1e-7)
    assert not position_encoding(50,width)[0].any()


def test_namespace_does_not_import_optional_ml():
    script='''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('torch','sklearn'): raise RuntimeError(fullname)
sys.meta_path.insert(0,Block())
import topokit
from topokit.workflows.topoformer import TopoFormerConfig, tokenize
assert 'torch' not in sys.modules and 'sklearn' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',script],check=True,env=dict(os.environ,PYTHONPATH=str(Path(__file__).parents[1]/'src')))


def test_invalid_model_dimensions_rejected():
    with pytest.raises(ValueError):TopoFormerConfig(d_model=130)
    with pytest.raises(ValueError):TopoFormerConfig(d_model=128,num_heads=3)


def test_model_initialization_and_parameter_count():
    torch=pytest.importorskip('torch')
    from topokit.workflows.topoformer import build_model
    torch.manual_seed(0)
    model=build_model()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==3366913
    assert 'positions' in dict(model.named_buffers())
    assert not torch.equal(model.blocks[0].qkv.weight,model.blocks[1].qkv.weight)
    assert model.blocks[0].qkv.weight.data_ptr()!=model.blocks[1].qkv.weight.data_ptr()
    model.eval()
    x=torch.randn(2,10,50,55)
    with torch.inference_mode():
        y=model(x);y2=model(x)
    assert y.shape==(2,) and torch.isfinite(y).all()
    torch.testing.assert_close(y,y2,rtol=0,atol=0)


def test_training_scaler_excludes_holdout_and_bundle_roundtrip(tmp_path):
    torch=pytest.importorskip('torch')
    sklearn=pytest.importorskip('sklearn')
    from sklearn.preprocessing import StandardScaler
    from topokit.workflows.topoformer import build_model,save_bundle,load_predictor
    from topokit.workflows.topoformer.bundle import standardize,sha256
    torch.set_num_threads(2)
    rng=np.random.default_rng(4)
    train=rng.normal(size=(12,2*5*3)).astype(np.float32)
    holdout=np.full((3,30),50,dtype=np.float32)
    train[:,0]=7;holdout[:,0]=7
    scaler=StandardScaler().fit(train)
    np.testing.assert_array_equal(standardize(holdout,scaler.mean_,scaler.scale_),scaler.transform(holdout))
    assert np.max(scaler.mean_)<=7 and scaler.n_samples_seen_==12
    c=TopoFormerConfig(n_statistics=2,n_scales=5,n_channels=3,d_model=16,num_layers=2,num_heads=4)
    model=build_model(c).eval()
    schema=tmp_path/'schema.json';schema.write_text('{"test_shape":[2,5,3]}\n')
    save_bundle(tmp_path/'model',model,mean=scaler.mean_,scale=scaler.scale_,n_samples_seen=12,schema_path=schema,metadata={'test':True})
    predictor=load_predictor(tmp_path/'model')
    with torch.inference_mode():
        expected=model(torch.from_numpy(scaler.transform(holdout).reshape(-1,2,5,3))).numpy()
    actual=predictor.predict(holdout.reshape(-1,2,5,3),schema_sha256=sha256(schema))
    np.testing.assert_array_equal(expected,actual)
    with pytest.raises(ValueError,match='schema'):
        predictor.predict(holdout.reshape(-1,2,5,3),schema_sha256='incorrect')
    with (tmp_path/'model/weights.pt').open('ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='checksum'):load_predictor(tmp_path/'model')


def test_tiny_supervised_overfit():
    torch=pytest.importorskip('torch')
    from topokit.workflows.topoformer import build_model
    torch.set_num_threads(2);torch.manual_seed(18)
    c=TopoFormerConfig(n_statistics=2,n_scales=5,n_channels=3,d_model=32,num_layers=2,num_heads=4)
    model=build_model(c)
    x=torch.randn(16,2,5,3)
    target=2*x[:,0,0,0]+x[:,1,4,2]
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=1e-3)
    model.eval()
    with torch.no_grad():initial=torch.mean((model(x)-target)**2).item()
    for _ in range(240):
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=torch.mean((model(x)-target)**2);loss.backward();optimizer.step()
    model.eval()
    with torch.no_grad():final=torch.mean((model(x)-target)**2).item()
    assert final<initial*0.08,(initial,final)
