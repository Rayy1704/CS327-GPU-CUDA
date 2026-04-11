#include "aes.h"
__constant__ unsigned char sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};
__constant__ unsigned char d_const_expanded_keys[176]; // 176 bytes for AES-128 expanded keys
__constant__ unsigned char mix_matrix[4][4] = {
    {0x02, 0x03, 0x01, 0x01},
    {0x01, 0x02, 0x03, 0x01},
    {0x01, 0x01, 0x02, 0x03},
    {0x03, 0x01, 0x01, 0x02}
};

__device__ void sub_bytes(aes_state*state){
    for(int r=0;r<4;r++){
        for(int c=0;c<4;c++){
            state->matrix[r][c]=sbox[state->matrix[r][c]]; // Select the byte from the sbox using the current byte as an index
        }
    }
}

__device__ void shift_rows(aes_state* state) {
    unsigned char temp;

    // Row 1: Left shift by 1 ([0,1,2,3] -> [1,2,3,0])
    temp = state->matrix[1][0];
    state->matrix[1][0] = state->matrix[1][1];
    state->matrix[1][1] = state->matrix[1][2];
    state->matrix[1][2] = state->matrix[1][3];
    state->matrix[1][3] = temp;

    // Row 2: Left shift by 2 ([0,1,2,3] -> [2,3,0,1])
    unsigned char t0 = state->matrix[2][0];
    unsigned char t1 = state->matrix[2][1];
    state->matrix[2][0] = state->matrix[2][2];
    state->matrix[2][1] = state->matrix[2][3];
    state->matrix[2][2] = t0;
    state->matrix[2][3] = t1;

    // Row 3: Left shift by 3 (Equivalent to Right shift by 1)
    temp = state->matrix[3][3];
    state->matrix[3][3] = state->matrix[3][2];
    state->matrix[3][2] = state->matrix[3][1];
    state->matrix[3][1] = state->matrix[3][0];
    state->matrix[3][0] = temp;
}

__device__ void mixcolumns(aes_state*state){
    for (int i=0;i<4;i++){
        unsigned char col[4];
        for (int j=0;j<4;j++){
            col[j]=state->matrix[j][i]; // Extract the current column from the state matrix
        }
        
        for(int j=0;j<4;j++){
            state->matrix[j][i]=
             multiply(mix_matrix[j][0],col[0])
            ^multiply(mix_matrix[j][1],col[1])
            ^multiply(mix_matrix[j][2],col[2])
            ^multiply(mix_matrix[j][3],col[3]); // Multiply the current column by the mix matrix and XOR it back into the state
        }
    }
}

__global__ void aes_encrypt_kernel(unsigned char * data, size_t lensize_t len){
    int idx=blockIdx.x*blockDim.x+threadIdx.x; // Calculate the global thread index
    int offset=idx*16; // Each thread processes a 16-byte block of data
    if(offset<len){ // Ensure we don't go out of bounds
        aes_state state;
        populate_state(&state, data+t); // Populate the state matrix with the input data (16 bytes for AES-128)
        // Initial AddRoundKey
        add_round_key(&state, d_const_expanded_keys, 0);
        // 9 main rounds of AES (SubBytes, ShiftRows, MixColumns, AddRoundKey)
        for(int i=0;i<9;i++){
            sub_bytes(&state); //Subbytes transformation
            shift_rows(&state); //ShiftRows transformation
            mixcolumns(&state); //MixColumns transformation
            add_round_key(&state,d_const_expanded_keys,i+1); //AddRoundKey transformation with the next round key
        }
        // Final round (without MixColumns)
        sub_bytes(&state); //Subbytes transformation
        shift_rows(&state); //ShiftRows transformation
        add_round_key(&state,d_const_expanded_keys,10); //AddRoundKey transformation with the final round key (round 10)
        // Copy the encrypted state back to the data buffer
        for (int c = 0; c < 4; c++) {
            for (int r = 0; r < 4; r++) {
                data[t+c * 4 + r] = state.matrix[r][c]; // Copy the state matrix back to the data buffer in column-major order
            }
        }    
    }
}

void aes_encrypt(unsigned char * data,unsigned char * expanded_keys, size_t len){
    unsigned char * d_data;
    unsigned char * d_expanded_keys;
    size_t data_size=len*sizeof(unsigned char);
    size_t keys_size=176*sizeof(unsigned char); // 176 bytes for AES-128
    cudaMalloc(&d_data, data_size);
    cudaMemcpy(d_data, data, data_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_const_expanded_keys, expanded_keys, 176);
    int threads_per_block=256;
    int blocks=((len+16-1)/16)/threads_per_block; // Calculate the number of blocks needed to process all data
    aes_encrypt_kernel<<<blocks, threads_per_block>>>(d_data,len); // Launch the AES encryption kernel on the GPU
    cudaMemcpy(data, d_data, data_size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_expanded_keys);
}
