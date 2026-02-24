
NVFLAGS = -O3 -Xcompiler "-Wall -Wextra -Werror"
APP = main

SRCS = main.cu matrix_io.cu matrix_ops.cu ci.cu matrix.cu

all:
	@nvcc $(NVFLAGS) $(SRCS) -o $(APP)
	@echo "Build successful! Run with: ./$(APP) <input_file> [output_file]"
clean:
	@rm -f $(APP)