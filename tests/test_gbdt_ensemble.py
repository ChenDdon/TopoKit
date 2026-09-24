"""Inference oracles for prediction averaging, embedded scaling and integrity."""
import csv
import json
import platform
import numpy as np
import pytest
from topokit.workflows.protein_ligand_prediction import prediction as api
from topokit.workflows.protein_ligand_prediction import features as fs


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


@pytest.fixture
def bundle(tmp_path):
    sklearn=pytest.importorskip('sklearn');joblib=pytest.importorskip('joblib');scipy=pytest.importorskip('scipy')
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  sklearn=sklearn.__version__,joblib=joblib.__version__)
    features=tmp_path/'features';features.mkdir();(features/'samples').mkdir();(features/'records').mkdir()
    write(features/'feature_schema.json',fs.schema());schema_hash=api.sha256(features/'feature_schema.json')
    rng=np.random.default_rng(2);train=np.zeros((32,27500),dtype=np.float32)
    train[:,:4]=rng.normal(20,3,(32,4));y=train[:,0]+train[:,1]**2*.1
    raw=train[:5].copy();ids=[f'x{i}' for i in range(5)]
    for name,row in zip(ids,raw):
        file=features/'samples'/(name+'.npy');np.save(file,row.reshape(10,50,55))
        write(features/'records'/(name+'.json'),dict(sample_id=name,recipe_id=fs.schema()['recipe_id'],
              status='success',output=dict(sha256=api.sha256(file))))
    manifest=tmp_path/'ids.csv';manifest.write_text('pdb_id\n'+'\n'.join(ids)+'\n')
    models=tmp_path/'models';models.mkdir();pipelines=[];members=[]
    for seed in range(3):
        folder=models/f'seed_{seed}';folder.mkdir()
        pipeline=Pipeline([('scaler',StandardScaler()),('gbdt',GradientBoostingRegressor(
            n_estimators=3,max_depth=2,subsample=.5,random_state=seed))]).fit(train,y)
        pipelines.append(pipeline);joblib.dump(pipeline,folder/'pipeline.joblib')
        scaler=pipeline['scaler'];np.savez(folder/'scaler.npz',mean=scaler.mean_,var=scaler.var_,scale=scaler.scale_)
        write(folder/'feature_schema.json',fs.schema())
        write(folder/'MODEL.json',dict(state='verified',software=versions,pipeline_contains_fitted_scaler=True,
            training_set='test_training',seed=seed,recipe_id=fs.schema()['recipe_id'],feature_schema_sha256=schema_hash,
            parameters=pipeline['gbdt'].get_params(),artifact_sha256={name:api.sha256(folder/name)
                for name in ('pipeline.joblib','scaler.npz','feature_schema.json')}))
        members.append(dict(seed=seed,path=f'seed_{seed}',model_receipt_sha256=api.sha256(folder/'MODEL.json')))
    write(models/'ENSEMBLE.json',dict(format='topokit.gbdt_ensemble.v1',state='verified',training_set='test_training',
        recipe_id=fs.schema()['recipe_id'],feature_schema_sha256=schema_hash,software=versions,
        aggregation='arithmetic_mean',seeds=[0,1,2],weights=[1/3]*3,members=members))
    return dict(root=tmp_path,models=models,features=features,manifest=manifest,raw=raw,pipelines=pipelines,versions=versions)


def output_values(path):
    with path.open() as f:return np.array([float(r['predicted_pK']) for r in csv.DictReader(f)])


def test_consensus_predicts_with_each_embedded_scaler_once(bundle):
    b=bundle;out=b['root']/'predictions.csv'
    result=api.predict(b['models'],b['features'],b['manifest'],out,batch_size=2)
    expected=np.mean([p.predict(b['raw']) for p in b['pipelines']],axis=0)
    np.testing.assert_array_equal(output_values(out),expected)
    double=np.mean([p.predict(p['scaler'].transform(b['raw'])) for p in b['pipelines']],axis=0)
    assert not np.allclose(expected,double)
    assert result['ensemble_size']==3 and result['seeds']==[0,1,2]
    with pytest.raises(FileExistsError):api.predict(b['models'],b['features'],b['manifest'],out)


def test_original_single_pipeline_still_supported(bundle):
    b=bundle;out=b['root']/'single.csv'
    result=api.predict(b['models']/'seed_0',b['features'],b['manifest'],out)
    np.testing.assert_array_equal(output_values(out),b['pipelines'][0].predict(b['raw']))
    assert result['ensemble_size']==1 and result['seeds'] is None


@pytest.mark.parametrize('field,value',[('seeds',[0,0,2]),('weights',[.5,.25,.25]),('aggregation','median')])
def test_rejects_changed_consensus_policy(bundle,field,value):
    b=bundle;file=b['models']/'ENSEMBLE.json';receipt=api.read(file);receipt[field]=value;write(file,receipt)
    with pytest.raises(ValueError,match='complete equal-weight'):api.predict(b['models'],b['features'],b['manifest'],b['root']/'bad.csv')


def test_rejects_missing_member_and_tampered_scaler(bundle):
    b=bundle;folder=b['models']/'seed_1';scaler=folder/'scaler.npz'
    scaler.write_bytes(scaler.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):api.predict(b['models'],b['features'],b['manifest'],b['root']/'bad.csv')
    (folder/'MODEL.json').unlink()
    with pytest.raises(FileNotFoundError):api.predict(b['models'],b['features'],b['manifest'],b['root']/'bad.csv')
    assert not (b['root']/'bad.csv').exists()


def test_rejects_relabelled_training_membership(bundle):
    b=bundle;folder=b['models']/'seed_2';file=folder/'MODEL.json';receipt=api.read(file)
    receipt['training_set']='different_training';write(file,receipt)
    e=api.read(b['models']/'ENSEMBLE.json');e['members'][2]['model_receipt_sha256']=api.sha256(file);write(b['models']/'ENSEMBLE.json',e)
    with pytest.raises(ValueError,match='identity mismatch'):api.predict(b['models'],b['features'],b['manifest'],b['root']/'bad.csv')
