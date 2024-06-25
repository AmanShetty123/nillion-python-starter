from nada_dsl import *

def nada_main():
    party1 = Party(name="Party1")
    my_int1 = SecretInteger(Input(name="my_int1", party=party1))  # This will be the message
    my_int2 = SecretInteger(Input(name="my_int2", party=party1))  # This will be the key

    # Encrypt the message
    encrypted_message = my_int1 ^ my_int2

    # Decrypt the message
    decrypted_message = encrypted_message ^ my_int2

    # Outputs for encrypted and decrypted messages
    encrypted_output = Output(encrypted_message, "encrypted_output", party1)
    decrypted_output = Output(decrypted_message, "decrypted_output", party1)

    return [encrypted_output, decrypted_output]

