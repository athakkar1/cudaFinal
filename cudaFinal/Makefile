# Compiler
NVCC := nvcc

# Compiler flags
CFLAGS := -g -G -I./Common

# Source files
SRC := audio.cu

# Output executable
OUT := audio

# Make all target
all: $(OUT)

# Target for building the executable
$(OUT): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<

# Clean target
clean:
	rm -f $(OUT)
