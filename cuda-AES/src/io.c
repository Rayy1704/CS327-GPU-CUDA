#include "io.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>


// Helper to get the size of a file using stat
long get_file_size_stat(const char *filename) {
    struct stat st;
    if (stat(filename, &st) == 0) {
        return st.st_size; // The size in bytes
    }
    return -1;
}
// Helper to convert a single hex char to its integer value
int hex_to_int(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1; // Not a hex char
}
unsigned char* read_file(const char* fileName, size_t *out_len){
    if (out_len != NULL) {
        *out_len = 0;
    }

    long size = get_file_size_stat(fileName);// Get file size using stat
    if (size == -1) {
        perror("Error getting file size");
        return NULL;
    }
    size_t buf_size = (size / 3); // Each byte is represented by 2 hex chars + 1 space, plus one for safety
    unsigned char * buf=malloc(buf_size);// Create a buffer to hold the binary data 
    if(buf == NULL){
        perror("Buffer Memory allocation failed");
        return NULL;
    }
    FILE *fp = fopen(fileName, "r");// Open the file for reading
    if (fp == NULL) {
        perror("Error opening file");
        free(buf);
        return NULL;
    }
    char *temp_text = malloc(size + 1);
    if (temp_text == NULL) {
        perror("Text buffer Memory allocation failed");
        fclose(fp);
        free(buf);
        return NULL;
    }
    size_t bytes_read = fread(temp_text, 1, size, fp);
    if (bytes_read != (size_t)size) {
        perror("Error reading file");
        fclose(fp);
        free(temp_text);
        free(buf);
        return NULL;
    }
    temp_text[size] = '\0';
    fclose(fp);
    
    size_t byte_count = 0; // Counter for the number of bytes read
    int high_nibble = -1; // Temporary variable to hold the high nibble of a byte

    // 3. The Parsing Loop
    for (long i = 0; i < size; i++) {
        int val = hex_to_int(temp_text[i]);
        if (val != -1) {
            if (high_nibble == -1) {
                high_nibble = val; // This is the first digit (e.g., the '5' in '53')
            } else {
                // This is the second digit (e.g., the '3' in '53')
                buf[byte_count++] = (high_nibble << 4) | val;
                high_nibble = -1; // Reset for next byte
            }
        }
    }
    free(temp_text);
    if (out_len != NULL) {
        *out_len = byte_count;
    }
    return buf;
}

void write_buf_to_file(const unsigned char *buf, size_t len, const char *out_filename) { // Write the binary data in buf to a file as hex string
    FILE *fp = fopen(out_filename, "w");
    if (fp == NULL) {
        perror("Error opening file for writing");
        return;
    }
    for (size_t i = 0; i < len; i++) {
        fprintf(fp, "%02x ", buf[i]);
    }
    fclose(fp);
}
unsigned char * read_key_file(const char *keyFileName) {
    FILE *fd = fopen(keyFileName, "r");
    if (fd == NULL) {
        perror("Error opening key file");
        return NULL;
    }
    unsigned char *temp_buf = malloc(49); // Allocate a buffer to hold the key data (48 bytes for 256-bit key)
    if (temp_buf == NULL) {
        perror("Key buffer Memory allocation failed");
        fclose(fd);
        return NULL;
    }
    size_t bytes_read = fread(temp_buf, 1, 48, fd); // Read up to 48 bytes from the key file
    if (bytes_read!=48) {
        perror("Error reading key file");
        free(temp_buf); // Free the allocated buffer
        fclose(fd);
        return NULL;    
    }
    temp_buf[48] = '\0'; // Null-terminate the buffer to ensure it's a valid string
    fclose(fd); // Close the file after reading
    
    size_t byte_count = 0; // Counter for the number of bytes read
    int high_nibble = -1; // Temporary variable to hold the high nibble of a byte
    unsigned char *initial_key = malloc(16); // Allocate memory for expanded keys
    if (initial_key == NULL) {
        perror("Initial key buffer allocation failed");
        free(temp_buf);
        return NULL;
    }
    // 3. The Parsing Loop
    for (long i = 0; i < 48; i++) {
        int val = hex_to_int(temp_buf[i]);
        if (val != -1) {
            if (high_nibble == -1) {
                high_nibble = val; // This is the first digit (e.g., the '5' in '53')
            } else {
                // This is the second digit (e.g., the '3' in '53')
                initial_key[byte_count++] = (high_nibble << 4) | val;
                high_nibble = -1; // Reset for next byte
            }
        }
    }
    free(temp_buf);
    return initial_key;
}