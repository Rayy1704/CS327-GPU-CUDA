#ifndef AES_H
#define AES_H

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

typedef struct {// AES state is a 4x4 matrix of bytes
    unsigned char matrix[4][4];
} aes_state;


// Function prototypes for AES operations

void aes_encrypt(unsigned char *input, unsigned char *expanded_keys, size_t data_len);// Encrypts the input data using the expanded keys

unsigned char* key_expansion(const unsigned char *key);// Expands the original key into multiple round keys for AES encryption

unsigned char xtime(unsigned char x);// Multiplies a byte by 2 in GF(2^8)

unsigned char multiply(unsigned char a, unsigned char b);// Multiplies two bytes in GF(2^8) using the xtime function

void populate_state(aes_state * state, unsigned char * input);// Populates the AES state matrix with input data

void add_round_key(aes_state * state, unsigned char * expanded_keys, int round);// Adds the round key to the state matrix for a given round

void aes_decrypt(unsigned char *input, unsigned char *expanded_keys, size_t data_len);// Decrypts the input data using the expanded keys

#endif // AES_H