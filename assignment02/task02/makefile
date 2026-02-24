
FLAGS = -Wall -Wextra -Werror -O3

# The name of your output program
APP = main

# List of your source files
SRCS = main.c matrix_io.c matrix_ops.c

all:
	@gcc $(FLAGS) $(SRCS) -o $(APP)
	@echo "Build successful! Run with: ./$(APP) <input_file> [output_file]"

clean:
	rm -f $(APP)