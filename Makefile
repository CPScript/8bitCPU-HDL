# Compiler settings
RUSTC = rustc
NASM = nasm
LD = ld

# Directories
KERNEL_DIR = kernel
BOOTLOADER_DIR = bootloader
BUILD_DIR = target
ISO_DIR = $(BUILD_DIR)/iso

# Flags
RUSTFLAGS = -C target-cpu=x86-64 -C target-feature=+sse,+sse2
NASMFLAGS = -f elf64
LDFLAGS = -n -T linker.ld

# Files
KERNEL_BIN = $(BUILD_DIR)/kernel.bin
BOOTLOADER_BIN = $(BUILD_DIR)/boot.bin
ISO_IMAGE = $(BUILD_DIR)/kernel-os.iso

.PHONY: all clean iso run

all: $(ISO_IMAGE)

$(ISO_IMAGE): $(KERNEL_BIN) $(BOOTLOADER_BIN)
	@mkdir -p $(ISO_DIR)/boot/grub
	cp $(KERNEL_BIN) $(ISO_DIR)/boot/
	cp $(BOOTLOADER_BIN) $(ISO_DIR)/boot/
	cp grub.cfg $(ISO_DIR)/boot/grub/
	grub-mkrescue -o $(ISO_IMAGE) $(ISO_DIR)

$(KERNEL_BIN):
	cd $(KERNEL_DIR) && cargo build --release
	cp $(KERNEL_DIR)/target/x86_64/release/kernel $(KERNEL_BIN)

$(BOOTLOADER_BIN): $(BOOTLOADER_DIR)/boot.asm
	@mkdir -p $(BUILD_DIR)
	$(NASM) $(NASMFLAGS) -o $(BOOTLOADER_BIN) $<

clean:
	rm -rf $(BUILD_DIR)
	cd $(KERNEL_DIR) && cargo clean

run: $(ISO_IMAGE)
	qemu-system-x86_64 -cdrom $(ISO_IMAGE) -m 256M -serial stdio

debug: $(ISO_IMAGE)
	qemu-system-x86_64 -cdrom $(ISO_IMAGE) -m 256M -serial stdio -s -S
