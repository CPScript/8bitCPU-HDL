; Basic bootloader for now (x86)
bits 16
org 0x7c00

boot:
    ; Initialize segment registers
    xor ax, ax
    mov ds, ax
    mov es, ax
    mov ss, ax
    mov sp, 0x7C00

    ; Enable A20 line
    in al, 0x92
    or al, 2
    out 0x92, al

    ; Load GDT
    cli
    lgdt [gdt_descriptor]

    ; Switch to protected mode
    mov eax, cr0
    or eax, 1
    mov cr0, eax

    ; Jump to 32-bit code
    jmp CODE_SEG:protected_mode

gdt_start:
    dq 0x0000000000000000   ; Null descriptor
gdt_code:
    dw 0xFFFF               ; Limit
    dw 0x0000               ; Base (low)
    db 0x00                 ; Base (middle)
    db 10011010b           ; Access byte
    db 11001111b           ; Flags + Limit (high)
    db 0x00                ; Base (high)
gdt_data:
    dw 0xFFFF
    dw 0x0000
    db 0x00
    db 10010010b
    db 11001111b
    db 0x00
gdt_end:

gdt_descriptor:
    dw gdt_end - gdt_start - 1
    dd gdt_start

CODE_SEG equ gdt_code - gdt_start
DATA_SEG equ gdt_data - gdt_start

bits 32
protected_mode:
    ; Set up segment registers
    mov ax, DATA_SEG
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Set up stack
    mov esp, 0x90000

    ; Load kernel
    mov eax, 1          ; Start from sector 1
    mov ecx, 100        ; Number of sectors to read
    mov edi, 0x100000   ; Load kernel at 1MB
    call load_kernel

    ; Jump to kernel
    jmp 0x100000

load_kernel:
    pusha
    push es

    mov bx, 0x1000      ; Load to ES:BX = 0x1000:0
    mov es, bx
    xor bx, bx

    .load_loop:
        push ax
        push cx

        ; Convert LBA to CHS
        mov dx, 0
        div word [sectors_per_track]
        inc dl          ; Sector = (LBA % SPT) + 1
        mov cl, dl      ; CL = sector number
        
        mov dx, 0
        div word [heads]
        mov dh, dl      ; DH = head
        mov ch, al      ; CH = cylinder

        mov dl, [boot_drive]
        mov ax, 0x0201  ; Read 1 sector
        int 0x13
        jc .disk_error

        pop cx
        pop ax

        add bx, 512     ; Advance buffer
        inc ax          ; Next sector
        loop .load_loop

    pop es
    popa
    ret

.disk_error:
    mov si, disk_error_msg
    call print_string
    jmp $

print_string:
    pusha
    mov ah, 0x0E
.loop:
    lodsb
    or al, al
    jz .done
    int 0x10
    jmp .loop
.done:
    popa
    ret

; Data
boot_drive:      db 0
sectors_per_track: dw 18
heads:           dw 2
disk_error_msg:  db "Disk error!", 0

times 510-($-$$) db 0
dw 0xAA55
