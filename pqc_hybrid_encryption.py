"""
Phase 2: Post-Quantum Secure Data Layer
----------------------------------------------------------------------
Secures the gene-expression CSV using ML-KEM (NIST FIPS 203 standard,
the finalized version of CRYSTALS-Kyber) + AES-256-GCM.

IMPORTANT CONCEPT: ML-KEM is a Key Encapsulation Mechanism (KEM), not a
general-purpose file cipher. It only wraps a small, fixed-size (32-byte)
shared secret -- it cannot directly encrypt an arbitrary-length CSV.

So the real pipeline is a HYBRID scheme (the same pattern Chrome uses for
post-quantum TLS today):
  1. ML-KEM securely agrees on a 32-byte shared secret between sender/receiver.
  2. That secret becomes the key for AES-256-GCM, which encrypts the actual
     file bytes.

Install once:
    pip install kyber-py cryptography
"""

import os
import json
import base64
import time
from kyber_py.ml_kem import ML_KEM_768          # NIST security level 3
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ---------- Step 1: Key generation (receiver side, e.g. the ML server) ----------
def generate_keypair():
    """Returns (encapsulation_key, decapsulation_key) i.e. (public, private)."""
    return ML_KEM_768.keygen()


# ---------- Step 2: Encrypt the CSV (sender side, e.g. the hospital/data source) ----------
def encrypt_file(input_path: str, output_path: str, encapsulation_key: bytes):
    shared_secret, kem_ciphertext = ML_KEM_768.encaps(encapsulation_key)

    aesgcm = AESGCM(shared_secret)          # already 32 bytes -> AES-256
    nonce = os.urandom(12)

    with open(input_path, "rb") as f:
        plaintext = f.read()
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    payload = {
        "kem_ciphertext": base64.b64encode(kem_ciphertext).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f)

    return len(kem_ciphertext), len(ciphertext)


# ---------- Step 3: Decrypt the CSV (right before feeding the MLP) ----------
def decrypt_file(encrypted_path: str, output_path: str, decapsulation_key: bytes):
    with open(encrypted_path, "r") as f:
        payload = json.load(f)

    kem_ciphertext = base64.b64decode(payload["kem_ciphertext"])
    nonce = base64.b64decode(payload["nonce"])
    ciphertext = base64.b64decode(payload["ciphertext"])

    shared_secret = ML_KEM_768.decaps(decapsulation_key, kem_ciphertext)

    aesgcm = AESGCM(shared_secret)
    plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)

    with open(output_path, "wb") as f:
        f.write(plaintext)


# ---------- Demo + benchmark (this is your Phase 4 "overhead" data) ----------
if __name__ == "__main__":
    ek, dk = generate_keypair()

    t0 = time.perf_counter()
    kem_ct_size, aes_ct_size = encrypt_file(
        "gene_expression.csv", "gene_expression.enc.json", ek
    )
    t1 = time.perf_counter()
    decrypt_file("gene_expression.enc.json", "gene_expression_decrypted.csv", dk)
    t2 = time.perf_counter()

    print(f"ML-KEM-768 ciphertext size : {kem_ct_size} bytes")
    print(f"AES-GCM ciphertext size    : {aes_ct_size} bytes")
    print(f"Encryption time            : {(t1 - t0) * 1000:.3f} ms")
    print(f"Decryption time            : {(t2 - t1) * 1000:.3f} ms")
    # Report these four numbers in your paper as the quantum-safety overhead.
