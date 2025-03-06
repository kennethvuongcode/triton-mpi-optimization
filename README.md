# Triton-Optimized MatMul Kernel and MPI-Based Parallel Training
In this project, I implemented a high-performance matrix multiplication kernel using Triton, optimized for execution on NVIDIA T4 GPUs. The kernel computes D = ReLU(A × B + C) by leveraging shared memory tiling, register tiling, and operator fusion to maximize efficiency. I fine-tuned block sizes and explored hyperparameter grid search to optimize memory access patterns and achieve significant speedup over a baseline PyTorch implementation.

Additionally, I implemented distributed communication primitives using MPI (Message Passing Interface) to support data-parallel and tensor-parallel training in a transformer model. I developed custom implementations of All-Reduce and All-to-All communication, tested their performance against native MPI functions, and implemented the necessary forward and backward communication protocols for distributed training.

**Key Concepts Implemented in this Project:**

- Triton-Based Matrix Multiplication Kernel: Implemented an optimized MatMul kernel with tiling, shared memory fetching, and register-level accumulation.
- Operator Fusion: Fused ReLU and element-wise addition to reduce memory traffic and improve efficiency.
- Hyperparameter Tuning: Performed grid search on block sizes to achieve maximum speedup.
- MPI Communication Primitives: Implemented custom All-Reduce and All-to-All operations for distributed workloads.
- Data Parallelism: Implemented dataset sharding and distribution across multiple processes.
- Tensor Model Parallelism: Designed partitioned linear layers for model-parallel forward and backward passes in a transformer layer.

This project demonstrates optimizations at both the GPU level (via Triton) and the distributed level (via MPI), improving performance for large-scale matrix operations and distributed deep learning models. Performance benchmarking was conducted on an NVIDIA T4 GPU, and communication efficiency was analyzed on an 8-core MPI setup.

To run the optimized MatMul kernel and distributed training tests, execute the provided Jupyter notebook and MPI test scripts.
