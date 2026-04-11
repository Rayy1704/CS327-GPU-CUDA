NVCC = nvcc
NVCCFLAGS = -O3 -Xcompiler -Wall,-Wextra
SRC_DIR = src
INC_DIR = include
INPUT_DIR = input
OUTPUT_DIR = output
TARGET = aes

# Note the .cu extensions for the CUDA files, while io.c remains standard C
SRCS = $(SRC_DIR)/main.cu $(SRC_DIR)/aes_encryption.cu  $(SRC_DIR)/aes_core.cu $(SRC_DIR)/io.c
HDRS = $(INC_DIR)/aes.h $(INC_DIR)/io.h

.PHONY: all run clean

all: $(TARGET)

run: $(TARGET)
	@./$(TARGET) $(INPUT_DIR)/data.txt $(INPUT_DIR)/key.txt $(OUTPUT_DIR)/output.txt

$(TARGET): $(SRCS) $(HDRS)
	@$(NVCC) $(NVCCFLAGS) -I$(INC_DIR) -o $@ $(SRCS)

clean:
	@rm -f $(TARGET) $(OUTPUT_DIR)/output.txt