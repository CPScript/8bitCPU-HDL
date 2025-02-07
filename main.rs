/// still being worked on (1/6th complete). V-0.9.3
use std::fmt;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};

/// Enum representing different types of instructions for the 8-bit CPU.
#[derive(Debug, Clone)]
enum Instruction {
    And(String, String, String),  // AND operation
    Or(String, String, String),   // OR operation
    Xor(String, String, String),  // XOR operation
    Not(String, String),          // NOT operation
    Shl(String, String),          // Shift left
    Shr(String, String),          // Shift right
    Rol(String, String),          // Rotate left
    Ror(String, String),          // Rotate right
    Jmp(usize),                   // Unconditional jump
    Jz(usize),                     // Jump if zero flag is set
    Jnz(usize),                    // Jump if zero flag is not set
    Jc(usize),                     // Jump if carry flag is set
    Cmp(String, String),           // Compare two registers
    Set(String, String),           // Set register based on condition
    Push(String),                  // Push register to stack
    Pop(String),                   // Pop register from stack
    Call(usize),                   // Call subroutine
    Ret,                            // Return from subroutine
    Mul(String, String, String),    // Multiply two registers (16-bit support)
    Div(String, String, String),    // Divide two registers (16-bit support)
    Int(u8),                        // Interrupt call
    Iret,                           // Interrupt return
    LoadMem(String, usize),         // Load from memory
    StoreMem(String, usize),        // Store to memory
    LoadIO(String, usize),          // Load from I/O memory
    StoreIO(String, usize),         // Store to I/O memory
    DmaTransfer(usize, usize, usize), // Direct Memory Access transfer
}

/// Structure representing an 8-bit CPU with 16-bit ALU support.
struct CPU {
    registers: [u8; 8], // 8 general-purpose registers
    memory: [u8; 256],  // 256-byte RAM
    rom: [u8; 256],     // 256-byte ROM
    stack: Vec<u8>,     // Stack for PUSH/POP operations
    pc: usize,          // Program counter
    zero_flag: bool,    // Zero flag
    carry_flag: bool,   // Carry flag
    pipeline: Vec<Instruction>, // Instruction pipeline
    io_memory: [u8; 256], // Memory-mapped I/O
    uart_buffer: Vec<u8>, // UART buffer for serial communication
    gpio_state: u8,      // General Purpose I/O state
    timer_counter: u16,  // Timer counter for system timing
}

impl CPU {
    /// Creates a new CPU instance.
    fn new() -> Self {
        CPU {
            registers: [0; 8],
            memory: [0; 256],
            rom: [0; 256],
            stack: Vec::new(),
            pc: 0,
            zero_flag: false,
            carry_flag: false,
            pipeline: Vec::new(),
            io_memory: [0; 256],
            uart_buffer: Vec::new(),
            gpio_state: 0,
            timer_counter: 0,
        }
    }

    /// Executes an instruction.
    fn execute_instruction(&mut self, instr: Instruction) {
        match instr {
            Instruction::Mul(a, b, out) => {
                let a_val = self.get_register(&a) as u16;
                let b_val = self.get_register(&b) as u16;
                let result = a_val * b_val;
                self.set_register(&out, (result & 0xFF) as u8);
            }
            Instruction::Div(a, b, out) => {
                let a_val = self.get_register(&a) as u16;
                let b_val = self.get_register(&b) as u16;
                if b_val != 0 {
                    let result = a_val / b_val;
                    self.set_register(&out, (result & 0xFF) as u8);
                }
            }
            Instruction::LoadMem(reg, addr) => {
                self.set_register(&reg, self.memory[addr]);
            }
            Instruction::StoreMem(reg, addr) => {
                self.memory[addr] = self.get_register(&reg);
            }
            Instruction::LoadIO(reg, addr) => {
                self.set_register(&reg, self.io_memory[addr]);
            }
            Instruction::StoreIO(reg, addr) => {
                self.io_memory[addr] = self.get_register(&reg);
            }
            Instruction::DmaTransfer(src, dest, size) => {
                for i in 0..size {
                    self.memory[dest + i] = self.memory[src + i];
                }
            }
            _ => {}
        }
    }
}

/// Defines the instruction set encoding for the 8-bit CPU.
pub struct Assembler {
    instructions: HashMap<String, u8>,
}

impl Assembler {
    pub fn new() -> Self {
        let mut instructions = HashMap::new();
        instructions.insert("AND", 0x00);
        instructions.insert("OR", 0x01);
        instructions.insert("XOR", 0x02);
        instructions.insert("NOT", 0x03);
        instructions.insert("SHL", 0x04);
        instructions.insert("SHR", 0x05);
        instructions.insert("ROL", 0x06);
        instructions.insert("ROR", 0x07);
        instructions.insert("JMP", 0x08);
        instructions.insert("JZ", 0x09);
        instructions.insert("JNZ", 0x0A);
        instructions.insert("JC", 0x0B);
        instructions.insert("CMP", 0x0C);
        instructions.insert("SET", 0x0D);
        instructions.insert("PUSH", 0x0E);
        instructions.insert("POP", 0x0F);
        instructions.insert("CALL", 0x10);
        instructions.insert("RET", 0x11);
        instructions.insert("MUL", 0x12);
        instructions.insert("DIV", 0x13);
        instructions.insert("INT", 0x14);
        instructions.insert("IRET", 0x15);
        instructions.insert("LOADM", 0x16);
        instructions.insert("STOREM", 0x17);
        instructions.insert("LOADIO", 0x18);
        instructions.insert("STOREIO", 0x19);
        instructions.insert("DMAT", 0x1A);

        Self { instructions }
    }

    /// Assembles an assembly source file into machine code.
    pub fn assemble(&self, input: &str, output: &str) -> io::Result<()> {
        let mut file = File::open(input)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let mut machine_code: Vec<u8> = Vec::new();
        
        for line in contents.lines() {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() { continue; }
            
            if let Some(&opcode) = self.instructions.get(tokens[0]) {
                machine_code.push(opcode);
                for &token in &tokens[1..] {
                    if let Ok(value) = token.parse::<u8>() {
                        machine_code.push(value);
                    }
                }
            }
        }

        let mut out_file = File::create(output)?;
        out_file.write_all(&machine_code)?;
        Ok(())
    }
}

/// Loads machine code from a binary file into memory.
pub struct BinaryLoader;

impl BinaryLoader {
    pub fn load(filename: &str) -> io::Result<Vec<u8>> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(buffer)
    }
}

/// Represents the CPU with registers and memory.
pub struct CPU {
    registers: [u8; 8],
    memory: [u8; 256],
    pc: usize,
}

impl CPU {
    pub fn new() -> Self {
        CPU {
            registers: [0; 8],
            memory: [0; 256],
            pc: 0,
        }
    }

    /// Loads a program into memory.
    pub fn load_program(&mut self, program: Vec<u8>) {
        self.memory[..program.len()].copy_from_slice(&program);
    }

    /// Executes the loaded program.
    pub fn execute(&mut self) {
        while self.pc < self.memory.len() {
            let opcode = self.memory[self.pc];
            self.pc += 1;
            match opcode {
                0x00 => println!("Executing AND instruction"),
                0x01 => println!("Executing OR instruction"),
                0x02 => println!("Executing XOR instruction"),
                0x08 => println!("Executing JMP instruction"),
                _ => println!("Unknown instruction: {:#04X}", opcode),
            }
        }
    }
}

///fn main() {
///    let assembler = Assembler::new();
///    if let Err(e) = assembler.assemble("program.asm", "program.bin") {
///        eprintln!("Error: {}", e);
///    } else {
///        println!("Assembly successful!");
///    }
///
///    match BinaryLoader::load("program.bin") {
///        Ok(data) => {
///            println!("Loaded binary: {:?}", data);
///            let mut cpu = CPU::new();
///            cpu.load_program(data);
///            cpu.execute();
///        }
///        Err(e) => eprintln!("Failed to load binary: {}", e),
///    }
///}


/// CPU Simulator for debugging and cycle-accurate execution.
pub struct Simulator {
    cpu: CPU,
    breakpoints: Vec<usize>,
}

impl Simulator {
    pub fn new() -> Self {
        Simulator {
            cpu: CPU::new(),
            breakpoints: Vec::new(),
        }
    }

    pub fn load_program(&mut self, program: Vec<u8>) {
        self.cpu.load_program(program);
    }

    pub fn set_breakpoint(&mut self, address: usize) {
        self.breakpoints.push(address);
    }

    pub fn step(&mut self) {
        let opcode = self.cpu.memory[self.cpu.pc];
        println!("Executing instruction at PC: {:#04X}, Opcode: {:#04X}", self.cpu.pc, opcode);
        self.cpu.execute();
    }

    pub fn run(&mut self) {
        while self.cpu.pc < self.cpu.memory.len() {
            if self.breakpoints.contains(&self.cpu.pc) {
                println!("Hit breakpoint at {:#04X}", self.cpu.pc);
                break;
            }
            self.step();
        }
    }
}

/// Verilog Generator for FPGA compatibility.
pub struct VerilogGenerator;

impl VerilogGenerator {
    pub fn generate() -> String {
        let verilog_code = """
        module cpu (
            input clk,
            input reset,
            output reg [7:0] registers [7:0],
            output reg [7:0] memory [255:0]
        );
            reg [7:0] pc;
            initial begin
                pc = 0;
            end

            always @(posedge clk or posedge reset) begin
                if (reset)
                    pc <= 0;
                else begin
                    case (memory[pc])
                        8'h00: registers[0] <= registers[0] & registers[1]; // AND operation
                        8'h01: registers[0] <= registers[0] | registers[1]; // OR operation
                        8'h02: registers[0] <= registers[0] ^ registers[1]; // XOR operation
                        default: pc <= pc + 1;
                    endcase
                end
            end
        endmodule
        """.to_string();
        
        verilog_code
    }
}

///fn main() {
///    let verilog_output = VerilogGenerator::generate();
///    println!("Generated Verilog:\n{}", verilog_output);
///}

fn main() {
    let mut cpu = CPU::new();
    cpu.pipeline.push(Instruction::Mul("R1".to_string(), "R2".to_string(), "R3".to_string()));
    cpu.pipeline.push(Instruction::LoadMem("R4".to_string(), 10));
    cpu.pipeline_execute();
}
