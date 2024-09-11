.section .init
.global _start
.type   _start, @function

_start:
	.option push
	.option norelax
	// Initialize global pointer,
	// Use an absolute reloc, so text placement is irrelevant.
	lui	gp, %hi(__global_pointer$)
	addi	gp, gp, %lo(__global_pointer$)
	.option pop
	tail    _Z13kernel_launchv
	.size  _start, .-_start
