from nada_dsl import *
import random
from Crypto.Cipher import AES
import hashlib
import logging
import time
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def generate_random_secret(length: int) -> int:
    """Generate a random secret integer of specified length."""
    return random.getrandbits(length)

def aes_encrypt_decrypt(message: bytes, key: bytes, mode: str) -> bytes:
    """Encrypt or decrypt a message using AES."""
    cipher = AES.new(key, AES.MODE_EAX)
    if mode == 'encrypt':
        ciphertext, tag = cipher.encrypt_and_digest(message)
        return cipher.nonce + ciphertext
    elif mode == 'decrypt':
        nonce = message[:16]
        ciphertext = message[16:]
        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        return cipher.decrypt(ciphertext)

def hash_key(key: str) -> bytes:
    """Generate a SHA-256 hash of the key."""
    return hashlib.sha256(key.encode()).digest()

def nada_main():
    logging.info("Starting the complex NADA program...")

    party1 = Party(name="Party1")
    my_int1 = SecretInteger(Input(name="my_int1", party=party1))  # This will be the message
    my_int2 = SecretInteger(Input(name="my_int2", party=party1))  # This will be the key

    # Encrypt the message with XOR operation
    logging.info("Encrypting the message with XOR operation...")
    encrypted_message_xor = my_int1 ^ my_int2

    # Decrypt the message with XOR operation
    logging.info("Decrypting the message with XOR operation...")
    decrypted_message_xor = encrypted_message_xor ^ my_int2

    # Generate a random secret key
    logging.info("Generating a random secret key...")
    secret_key = generate_random_secret(128)  # 128-bit key

    # Convert the integer message and key to bytes for AES encryption
    message_bytes = my_int1.to_bytes((my_int1.bit_length() + 7) // 8, byteorder='big')
    key_bytes = secret_key.to_bytes((secret_key.bit_length() + 7) // 8, byteorder='big')

    # Encrypt the message with AES
    logging.info("Encrypting the message with AES...")
    encrypted_message_aes = aes_encrypt_decrypt(message_bytes, key_bytes, 'encrypt')

    # Decrypt the message with AES
    logging.info("Decrypting the message with AES...")
    decrypted_message_aes = aes_encrypt_decrypt(encrypted_message_aes, key_bytes, 'decrypt')

    # Convert decrypted bytes back to integer
    decrypted_message_aes_int = int.from_bytes(decrypted_message_aes, byteorder='big')

    # Hash the key for secure storage/transmission
    logging.info("Hashing the key for secure storage/transmission...")
    hashed_key = hash_key(str(my_int2))

    # Measure performance
    logging.info("Measuring performance...")
    start_time = time.time()
    for _ in range(1000):
        _ = my_int1 ^ my_int2
    xor_performance = time.time() - start_time

    start_time = time.time()
    for _ in range(1000):
        aes_encrypt_decrypt(message_bytes, key_bytes, 'encrypt')
    aes_performance = time.time() - start_time

    # Outputs for encrypted and decrypted messages
    encrypted_output_xor = Output(encrypted_message_xor, "encrypted_output_xor", party1)
    decrypted_output_xor = Output(decrypted_message_xor, "decrypted_output_xor", party1)
    encrypted_output_aes = Output(encrypted_message_aes, "encrypted_output_aes", party1)
    decrypted_output_aes = Output(decrypted_message_aes_int, "decrypted_output_aes", party1)
    hashed_key_output = Output(hashed_key, "hashed_key_output", party1)

    logging.info("NADA program completed successfully.")

    return [
        encrypted_output_xor, decrypted_output_xor,
        encrypted_output_aes, decrypted_output_aes,
        hashed_key_output
    ]

# Execute the main function
outputs = nada_main()
for output in outputs:
    logging.info(f"{output.name}: {output.value}")
