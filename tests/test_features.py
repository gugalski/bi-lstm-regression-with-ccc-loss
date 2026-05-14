import numpy as np
import pytest

from src.features import cache_exists, cache_path_for, load_cached


def test_cache_path_for(tmp_path):
    p = cache_path_for("001.wav", tmp_path)
    assert p == tmp_path / "001.npy"


def test_cache_exists_false(tmp_path):
    assert not cache_exists("001.wav", tmp_path)


def test_cache_exists_true(tmp_path):
    arr = np.zeros((10, 88), dtype=np.float32)
    np.save(tmp_path / "001.npy", arr)
    assert cache_exists("001.wav", tmp_path)


def test_load_cached(tmp_path):
    arr = np.random.randn(50, 88).astype(np.float32)
    np.save(tmp_path / "test.npy", arr)
    loaded = load_cached(tmp_path / "test.npy")
    assert loaded.dtype == np.float32
    assert loaded.shape == (50, 88)
    np.testing.assert_array_almost_equal(loaded, arr)
