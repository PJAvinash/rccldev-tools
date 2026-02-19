# ================================
# HIP Smoke Test Makefile
# ================================

# ---- Configurable variables ----
CXX           := hipcc            # hipcc (default) or amdclang++
TARGET        := hip_api_test
SRC           := hip_api_test.cpp
BUILD_DIR     := build
OBJ           := $(BUILD_DIR)/hip_api_test.o

# Build type: make BUILD=release
BUILD         ?= debug

# Optional GPU arch override:
# make GPU_ARCH=gfx90a
GPU_ARCH      ?=

# ---- Compiler flags ----
CXXFLAGS_DEBUG   := -O0 -g -std=c++17 -Wall -Wextra
CXXFLAGS_RELEASE := -O3 -std=c++17 -DNDEBUG

ifeq ($(BUILD),release)
    CXXFLAGS := $(CXXFLAGS_RELEASE)
else
    CXXFLAGS := $(CXXFLAGS_DEBUG)
endif

# ---- HIP specific handling ----
# If using amdclang++, we must add HIP flags manually
ifeq ($(CXX),amdclang++)
    HIPFLAGS := -x hip --offload-arch=$(GPU_ARCH)
else
    # hipcc automatically handles HIP
    ifneq ($(GPU_ARCH),)
        HIPFLAGS := --offload-arch=$(GPU_ARCH)
    endif
endif

# Auto dependency generation
DEPFLAGS = -MMD -MP
DEPS     := $(OBJ:.o=.d)

# ---- Default target ----
all: $(TARGET)

# ---- Build rules ----
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(OBJ): $(SRC) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(HIPFLAGS) $(DEPFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(HIPFLAGS) $^ -o $@

# ---- Utilities ----
run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

rebuild: clean all

print-%:
	@echo '$*=$($*)'

# ---- Include auto-generated deps ----
-include $(DEPS)

.PHONY: all clean rebuild run
