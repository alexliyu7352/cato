import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from r0.model import build_generator, parameter_count
from r0.data import PairedDataset,load_manifest,load_example
from r0.train import warp_previous_to_current,gradient_loss

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'samples/smoke/manifest.jsonl'

def test_generator_sizes():
    sizes=[parameter_count(build_generator('A')),parameter_count(build_generator('B'))]
    assert 10_000_000<=sizes[0]<=30_000_000
    assert 5_000_000<=sizes[1]<=15_000_000

def test_generator_shapes_and_input_contract():
    torch.set_num_threads(2)
    for route in ('A','B'):
        model=build_generator(route)
        x=torch.rand((1,15,64,96));x[:,7:14]=0;x[:,7]=1
        with torch.no_grad():y=model(x)
        assert y.shape==(1,3,64,96)
        assert torch.isfinite(y).all() and y.min()>=0 and y.max()<=1
        with pytest.raises(ValueError):model(x[:,:12])

def test_smoke_target_guard():
    with pytest.raises(PermissionError):PairedDataset(MANIFEST,split='train')
    ds=PairedDataset(MANIFEST,split='train',allow_smoke=True,route='A')
    assert len(ds)==9
    x,y=ds[0];assert x.shape[0]==15 and y.shape[0]==3
    assert torch.allclose(x[7:14].sum(0),torch.ones_like(x[0]))

def test_pre_stylized_B_does_not_copy_target():
    rows=load_manifest(MANIFEST)
    x,y=load_example(MANIFEST.parent,rows[0],route='B')
    assert (x[:3]-y).abs().mean()>.01

def test_3d_fields_nonempty():
    row=load_manifest(MANIFEST)[0]
    d=np.load(MANIFEST.parent/row['depth'])
    n=np.load(MANIFEST.parent/row['normal'])
    sem=np.asarray(Image.open(MANIFEST.parent/row['semantic']))
    assert d.ndim==2 and n.shape==(*d.shape,3)
    assert np.std(d)>.001 and np.any((n!=0))
    assert len(np.unique(sem))>=4

def test_temporal_warp_coordinates():
    previous=torch.arange(25,dtype=torch.float32).view(1,1,5,5)
    flow=torch.zeros((1,2,5,5));flow[:,0]=1
    mask=torch.ones(1,1,5,5)
    out=warp_previous_to_current(previous,flow,mask)
    assert torch.allclose(out[0,0,2,:4],previous[0,0,2,1:5],atol=1e-5)

def test_gradient_loss_finite():
    a=torch.rand(1,3,16,16)
    assert gradient_loss(a,a).item()==0

def test_temporal_dataset_and_gradient_step():
    from r0.data import TemporalPairs
    ds=TemporalPairs(MANIFEST,allow_smoke=True,patch=64)
    x,y,px,py,flow,valid=ds[0]
    assert x.shape==(15,64,64) and flow.shape==(2,64,64)
    assert valid.shape==(1,64,64)
    model=build_generator('A',base=16)
    out=model(x[None]);loss=gradient_loss(out,y[None]);loss.backward()
    assert any(p.grad is not None for p in model.parameters())

def test_quality_audit_rejects_leakage_and_synthetic_targets():
    from r0.audit import audit
    smoke=audit(MANIFEST)
    assert smoke['rows']==12
    with pytest.raises(RuntimeError):audit(MANIFEST,quality=True)

def test_teacher_ingest_requires_rights(tmp_path):
    from r0.teacher_ingest import ingest
    rights=tmp_path/'rights.json';rights.write_text(json.dumps({'teacher_model_id':'test','rights_reference':'test',
        'outputs_can_train_independent_model':False}))
    review=tmp_path/'review.jsonl';review.write_text('{}\n')
    with pytest.raises(PermissionError):ingest(MANIFEST,tmp_path,rights,review,tmp_path/'out.jsonl')
