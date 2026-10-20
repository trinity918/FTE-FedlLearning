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


def test_homomorphic_sum_does_not_modify_inputs(manager):
    a, b = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
    ea, eb = manager.encrypt(a), manager.encrypt(b)
    total = manager.homomorphic_sum([ea, eb])
    np.testing.assert_allclose(manager.decrypt(total), a + b, atol=1e-4)
    np.testing.assert_allclose(manager.decrypt(ea), a, atol=1e-4)     # first input intact
    np.testing.assert_allclose(manager.decrypt(eb), b, atol=1e-4)
    assert total[0] is not ea[0]


def test_public_context_cannot_decrypt(manager):
    import tenseal as ts

    public = ts.context_from(manager.public_context_bytes(galois_keys=False))
    assert not public.is_private() and not public.has_galois_keys()
    ct = ts.ckks_vector(public, [0.5, -0.5])
    with pytest.raises(ValueError):
        ct.decrypt()
    back = ts.ckks_vector_from(manager.context, ct.serialize())
    np.testing.assert_allclose(back.decrypt(), [0.5, -0.5], atol=1e-4)
    assert ts.context_from(manager.public_context_bytes()).has_galois_keys()
