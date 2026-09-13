import numpy as np
import pytest

from scripts.diagnose_conditioning import fit_scaler, transform


def test_training_scaler_does_not_refit_on_shifted_validation():
    train = np.array([[1, 7], [3, 7]], dtype=np.float32)
    mean, scale = fit_scaler(train)
    np.testing.assert_array_equal(mean, [2, 7])
    np.testing.assert_array_equal(scale, [1, 1])
    validation = np.array([[102, 8]], dtype=np.float32)
    np.testing.assert_array_equal(transform(validation, mean, scale), [[100, 1]])
    np.testing.assert_array_equal(transform(train, mean, scale), [[-1, 0], [1, 0]])
    with pytest.raises(ValueError):
        fit_scaler(np.array([[np.nan, 1]]))
