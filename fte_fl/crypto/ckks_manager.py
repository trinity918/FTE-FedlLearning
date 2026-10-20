"""Chunked CKKS vector operations using TenSEAL."""
from __future__ import annotations

import numpy as np
import tenseal as ts


class CKKSManager:
    def __init__(
        self,
        poly_modulus_degree: int = 16384,
        coeff_mod_bit_sizes: list[int] | None = None,
        global_scale: float = 2**40,
        chunk_size: int = 8192,
    ):
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]
        if chunk_size < 1 or chunk_size > poly_modulus_degree // 2:
            raise ValueError("chunk_size must be between 1 and poly_modulus_degree / 2")

        self.chunk_size = chunk_size
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )
        self.context.global_scale = global_scale
        self.context.generate_galois_keys()

    def encrypt(self, values: np.ndarray | list[float]) -> list:
        """Encrypt a 1-D vector as independently encrypted chunks."""
        vector = np.asarray(values, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError("Expected a one-dimensional vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Input contains NaN or infinite values")

        return [
            ts.ckks_vector(self.context, chunk.tolist())
            for start in range(0, len(vector), self.chunk_size)
            if len(chunk := vector[start : start + self.chunk_size]) > 0
        ]

    def public_context_bytes(self, galois_keys: bool = True) -> bytes:
        """Serialized copy of the context WITHOUT the secret key.

        It keeps the public and relinearization keys, and the Galois keys if
        `galois_keys` (needed for dot products; they dominate the size).
        Holders can encrypt and evaluate additions and plaintext-scalar
        products, but cannot decrypt.
        """
        public = self.context.copy()
        public.make_context_public()
        return public.serialize(save_galois_keys=galois_keys, save_relin_keys=True)

    def decrypt(self, encrypted_chunks: list) -> np.ndarray:
        """Decrypt chunks and concatenate them in their original order."""
        if not encrypted_chunks:
            return np.array([], dtype=np.float64)
        return np.concatenate(
            [np.asarray(chunk.decrypt(), dtype=np.float64) for chunk in encrypted_chunks]
        )

    def homomorphic_sum(self, encrypted_vectors: list[list]) -> list:
        """Sum equally sized encrypted vectors without decrypting them.

        The inputs are not modified (the running sum starts from a copy).
        """
        if not encrypted_vectors:
            raise ValueError("At least one encrypted vector is required")

        expected_chunks = len(encrypted_vectors[0])
        if any(len(vector) != expected_chunks for vector in encrypted_vectors):
            raise ValueError("Encrypted vectors have different chunk counts")

        summed = []
        for chunk_index in range(expected_chunks):
            chunks = [vector[chunk_index] for vector in encrypted_vectors]
            result = chunks[0].copy()
            for chunk in chunks[1:]:
                result += chunk
            summed.append(result)
        return summed
