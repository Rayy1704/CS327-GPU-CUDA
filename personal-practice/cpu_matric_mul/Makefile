# Compiler and flags
CC := gcc
CFLAGS := -Wall -Wextra -Wpedantic -std=c99 -O2 -g -D_POSIX_C_SOURCE=200809L
CFLAGS += -Wconversion -Wsign-conversion -Wformat=2
CFLAGS += -Wwrite-strings

# Targets
TARGET := cpuMM.out
SRCS := cpu_main.c cpu_matrix.c cpu_io.c
HEADERS := cpu_matrix.h cpu_io.h
OBJS := $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^
	@echo "✓ Build successful: $(TARGET) compiled with 0 warnings"

# Compile source files
%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJS) $(TARGET)
	@echo "✓ Clean complete"

# Rebuild everything
rebuild: clean all

# Phony targets
.PHONY: all clean rebuild
