import numpy as np
import torch
import pytest

from scripts.run_controlled_replication import geometry, models_for_seed
from scripts.evaluate_controlled_replication import summarize


def test_simple_geometry_uses_unordered_real_atoms_and_physical_scale():
    z = np.array([1, 6, 8])
    pos = np.array([[0,0,0], [3,0,0], [0,4,0]], dtype=np.float32)
    v = geometry(z,pos)
    np.testing.assert_allclose(v[:5],[1,1,0,1,0])
    np.testing.assert_allclose(v[5:],[4,np.std([3,4,5]),3,5,np.sqrt(50/3)],rtol=1e-6)
    np.testing.assert_allclose(geometry(z[::-1],pos[::-1, [1,2,0]]+9),v,rtol=1e-6)
    np.testing.assert_allclose(geometry(z,pos*2)[5:],v[5:]*2,rtol=1e-6)
    np.testing.assert_array_equal(geometry([1],[[0,0,0]])[5:],np.zeros(5))


def test_replication_initialization_is_shared_and_control_is_identical():
    models = models_for_seed(42)
    for model in models.values():
        for name,value in models['egnn'].encoder.state_dict().items():
            torch.testing.assert_close(model.encoder.state_dict()[name],value,rtol=0,atol=0)
        for name,value in models['egnn'].head.state_dict().items():
            torch.testing.assert_close(model.head.state_dict()[name],value,rtol=0,atol=0)
    for name,value in models['tda'].state_dict().items():
        torch.testing.assert_close(models['constant'].state_dict()[name],value,rtol=0,atol=0)


def test_replication_summary_requires_complete_paired_grid():
    rows=[dict(scope='noise_subset',sigma=0.,noise_seed=0,seed=s,arm=a,condition='matched',
               molecule_id=i,target_eV=0.,prediction_eV=.9 if a=='tda' else 1.)
          for s in (42,43,44) for a in ('egnn','tda','geometry','constant') for i in (1,2)]
    summary=summarize(rows,[42,43,44])
    assert summary['comparisons'][0]['mean_difference_eV']==pytest.approx(-.1)
    with pytest.raises(ValueError,match='Incomplete'):
        summarize(rows[:-1],[42,43,44])
    with pytest.raises(ValueError,match='Duplicate'):
        summarize(rows+[rows[0]],[42,43,44])
