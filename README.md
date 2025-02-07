# 8-bit CPU Simulator and Assembler

>This project implements a simple 8-bit CPU simulator with support for a 16-bit ALU and various hardware functionalities. The simulator includes a full assembler, a machine code loader, and debugging tools. The CPU supports both software execution and FPGA compatibility via Verilog generation.

## Features
- **8-bit CPU with 16-bit ALU support**: Implements basic arithmetic and logical operations.
- **Assembler**: Converts assembly language instructions into machine code.
- **Binary Loader**: Loads machine code from binary files into memory.
- **CPU Execution**: Executes instructions in a simulated environment.
- **Cycle-Accurate Simulator**: Provides step-by-step execution and debugging with breakpoints.
- **FPGA Verilog Code Generation**: Generates Verilog code for FPGA hardware implementation.

## Components

### `CPU` Struct
- Contains the CPU state, including general-purpose registers, memory, stack, program counter (PC), and status flags.
- Supports 8-bit arithmetic and logical operations, conditional jumps, memory operations, and more.

### `Instruction` Enum
Represents various types of instructions, including:
- Logical operations (`AND`, `OR`, `XOR`, `NOT`).
- Shifts and rotates (`SHL`, `SHR`, `ROL`, `ROR`).
- Jump operations (`JMP`, `JZ`, `JNZ`, `JC`).
- Stack operations (`PUSH`, `POP`).
- Memory and I/O operations (`LoadMem`, `StoreMem`, `LoadIO`, `StoreIO`).
- Arithmetic operations (`MUL`, `DIV`).
- DMA transfer operations.

### `Assembler` Struct
- Assembles assembly code into machine code.
- The instruction set is encoded in a `HashMap`, where each instruction has a corresponding opcode.

### `BinaryLoader` Struct
- Loads machine code from a binary file into memory.

### `Simulator` Struct
- A debugging tool for simulating and stepping through programs.
- Supports setting breakpoints and executing instructions one cycle at a time for debugging purposes.

### `VerilogGenerator` Struct
- Generates Verilog code for FPGA compatibility, including a simple CPU module that simulates the core instructions.

## How to Use

### Assemble Assembly Code
To assemble a `.asm` file into machine code, use the `Assembler` struct:
```rust
let assembler = Assembler::new();
assembler.assemble("program.asm", "program.bin").expect("Assembly failed");
```

### Load and Execute Machine Code
Once the machine code is generated, it can be loaded into the CPU's memory and executed:
```rust
let binary = BinaryLoader::load("program.bin").expect("Failed to load binary");
let mut cpu = CPU::new();
cpu.load_program(binary);
cpu.execute();
```

### Run the Simulator
For step-by-step debugging and cycle-accurate simulation:
```rust
let mut simulator = Simulator::new();
simulator.load_program(binary);
simulator.set_breakpoint(0x10); // Set a breakpoint at address 0x10
simulator.run(); // Execute the program until the breakpoint is hit
```

### Generate Verilog Code
Generate Verilog code for FPGA implementation:
```rust
let verilog_code = VerilogGenerator::generate();
println!("Generated Verilog code:\n{}", verilog_code);
```
### Future additions
## License
