import pytest
from scannet.scannet_loader import SceneDataset, _resolve_split_offset
from scannet.eval_rgb_psnr import _filter_kwargs_for_callable

@pytest.mark.parametrize('mode,offset,expected', [('llff',None,0),('llff',2,2),('scannet_feature3dgs',None,2),('replica_feature3dgs',None,2),('auto',None,2),('auto',0,0)])
def test_scannet_split(mode,offset,expected):
    assert _resolve_split_offset('/data/scene0059_00',mode,offset,8)==expected

def test_llff_default_is_preserved():
    assert _resolve_split_offset('/data/bonsai','llff',None,8)==0
    assert _resolve_split_offset('/data/bonsai','auto',None,8)==0

def test_evaluator_does_not_drop_explicit_split():
    kwargs={'split_mode':'scannet_feature3dgs','split_test_offset':2}
    assert _filter_kwargs_for_callable(SceneDataset.__init__,kwargs)==kwargs

def test_scannet_loader_defaults_to_feature3dgs_split():
    import inspect
    mode = inspect.signature(SceneDataset).parameters['split_mode'].default
    assert mode == 'scannet_feature3dgs'
    assert _resolve_split_offset('/data/scene0000_00',mode,None,8) == 2

@pytest.mark.parametrize('mode,offset,hold',[('typo',2,8),('llff',-1,8),('llff',8,8),('llff',0,0)])
def test_invalid_split_fails(mode,offset,hold):
    with pytest.raises(ValueError): _resolve_split_offset('/data/scene0059_00',mode,offset,hold)

def test_train_test_disjoint_complete():
    for offset in (0,2):
        test=set(range(_resolve_split_offset('/data/scene0059_00','llff',offset,8),200,8))
        train=set(range(200))-test
        assert len(test)==25 and len(train)==175
        assert train.isdisjoint(test) and train|test==set(range(200))
