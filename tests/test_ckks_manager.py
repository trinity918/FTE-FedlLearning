import numpy as np
import pytest

from fte_fl.crypto.ckks_manager import CKKSManager


@pytest.fixture(scope="module")
def manager():
    return CKKSManager()


def test_chunked_encrypt_decrypt_200k(manager):
    rng = np.random.default_rng(42)
    values = rng.uniform(-0.1, 0.1, size=200_000)

    encrypted = manager.encrypt(values)
    recovered = manager.decrypt(encrypted)

    assert len(encrypted) == 25
    assert recovered.shape == values.shape
    np.testing.assert_allclose(recovered, values, atol=1e-3, rtol=1e-3)


def test_homomorphic_sum_five_vectors(manager):
    rng = np.random.default_rng(123)
    vectors = [rng.uniform(-0.1, 0.1, size=200_000) for _ in range(5)]
    encrypted = [manager.encrypt(vector) for vector in vectors]

    encrypted_sum = manager.homomorphic_sum(encrypted)
    recovered = manager.decrypt(encrypted_sum)
    expected = np.sum(vectors, axis=0)

    np.testing.assert_allclose(recovered, expected, atol=1e-3, rtol=1e-3)


def test_rejects_non_finite_input(manager):
    with pytest.raises(ValueError):
        manager.encrypt([1.0, float("nan")])


def test_rejects_empty_sum(manager):
    with pytest.raises(ValueError):
        manager.homomorphic_sum([])
