# MiniMixtral: Hybrid State-Space MoE Foundation Model

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

**MiniMixtral**  is a high-performance, from-scratch foundation model architecture designed for resource-constrained pretraining environments (e.g., Kaggle Dual T4 GPUs). It fuses the long-context retrieval capabilities of traditional Transformers with the linear-time sequence modeling of State Space Models (Mamba), scaled via a Block-Sparse Mixture-of-Experts (MoE) routing paradigm.

To achieve 100% compute saturation and bypass severe I/O bottlenecks during training, this repository introduces a **Zero-Copy Asynchronous Data Engine** alongside a novel **Tri-Engine Hybrid Optimizer** that routes structural parameters to specialized mathematical backends (Muon, Sophia-G, and AdamW).

---

## 1. Architectural Innovations

The model diverges from standard dense Transformer topologies by implementing a periodic hybrid layer structure, heavily optimized for memory efficiency and throughput.

### 1.1 Hybrid Decoder Topology (`HybridDecoderLayer`)
Standard attention mechanisms scale quadratically with sequence length, making long-context training unviable on limited VRAM. Transmamba mitigates this by interleaving standard Attention with Mamba blocks:
* **Periodic Injection:** Every 4th layer in the network is replaced with a Mamba block (`use_mamba=(i % 4 == 0)`). 
* **Mechanism:** This allows the network to maintain precise, token-level retrieval via attention, while offloading long-range temporal state compression to the Mamba layers, significantly reducing the KV-cache memory footprint.
* **Sliding Window Attention (SWA):** Self-attention is restricted to a fixed local window (2048 tokens), combined with Rotary Positional Embeddings (RoPE) for relative positional awareness.

### 1.2 Block-Sparse Mixture-of-Experts (`SparseMoE`)
Instead of activating the full parameter space for every token, the model utilizes a capacity-constrained routing mechanism.
* **Top-K Dispatch:** A routing linear layer dispatches each token to the top 2 out of 8 available experts.
* **Vectorized Capacity Constraints:** To prevent hardware lockups, expert capacity is strictly enforced natively without Python loops, utilizing `cumsum` masking to drop tokens that exceed the hardware limit.
* **Auxiliary Objectives:** To ensure routing stability, two auxiliary losses are computed at every layer and added to the final cross-entropy objective:
  
  **1. Load Balancing Loss ($L_{aux}$):** Ensures uniform token distribution across the expert pool.
  $$L_{aux} = E \sum_{i=1}^{E} f_i \cdot P_i$$
  *(Where $E$ is the number of experts, $f_i$ is the fraction of tokens dispatched to expert $i$, and $P_i$ is the mean router probability for expert $i$.)*

  **2. Router Z-Loss ($L_z$):** Penalizes large logits to stabilize the softmax gradients and encourage routing exploration.
  $$L_z = \frac{1}{N} \sum_{i=1}^{N} \left( \log \sum_{j=1}^{E} \exp(x_{i,j}) \right)^2$$

---

## 2. Asynchronous Zero-Copy Data Engine

Pretraining on cloud environments like Kaggle often results in GPU starvation due to slow disk I/O. Transmamba abandons standard PyTorch `Dataset` paradigms for a highly parallel, zero-copy streaming engine.

### 2.1 The Pipeline Mechanics
* **Background Producer (`AsyncBinaryProducer`):** A daemon thread continuously pulls raw text, tokenizes it in the background, and serializes the integer IDs directly to disk as raw binary slabs (`.bin`). It maintains a rolling cache window (`max_chunks_ahead = 12`) to manage limited disk space dynamically.
* **Zero-Copy Consumer (`StreamingBinaryDataset`):** The training loop reads these binary chunks using `numpy.memmap`. This bypasses standard RAM allocations, mapping the disk blocks directly to virtual memory. The GPU streams data continuously without waiting for CPU batch collation.
* **Deterministic Fault Tolerance:** The dataset maintains granular pointers (`flushed_chunk_count`, `consumed_seq_in_chunk`). Upon hardware failure or Kaggle timeout, the dataloader resumes at the exact sequence index without token phase-shifting or data repetition.

---

## 3. Tri-Engine Hybrid Optimization

Different structural components of an LLM possess distinct loss landscapes and gradient variances. Transmamba employs a `TiedAwareParameterRouter` to dynamically map modules to three distinct optimization backends.

| Optimizer | Assigned Modules | Mathematical Justification |
| :--- | :--- | :--- |
| **Muon** | Embeddings, $V/O$ Projections, MoE MLPs, Mamba Projections | Employs Newton-Schulz orthogonalization iterations. Ideal for dense 2D structural matrices to maximize internal rank and representation capacity. |
| **Sophia-G** | $Q/K$ Projections, MoE Router Gates | A second-order optimizer utilizing diagonal Hessian estimates. Perfect for geometry-sensitive routing parameters where gradient variance is exceptionally high. |
| **AdamW** | RMSNorms, Biases, 1D/3D Tensors | Standard fallback for non-2D tensors and scale-invariant normalization layers where momentum and variance tracking are sufficient. |

### 3.1 Pretraining Orchestrator (`PretrainingOrchestrator`)
The hybrid setup is managed by an orchestrator that handles:
* **Tiered Gradient Clipping:** Global gradients are clipped at `1.0`, but volatile parameters are isolated (Experts clipped at `0.5`, Router gates at `0.1`).
* **Hessian Warmup:** A dedicated step-0 pass on real linguistic data to initialize Sophia's curvature tracking before weights are updated.
* **Exponential Moving Average (EMA):** Maintains a shadow copy of the weights (`ema_decay = 0.9999`) for improved inference stability post-training.

---

## 4. Configuration & Usage

### 4.1 System Requirements
* PyTorch 2.x+
* `transformers`, `numpy`
* Environment with at least 1 GPU (Optimized natively for `nn.DataParallel` on Kaggle Dual T4s).

### 4.2 Core Hyperparameters (`ModelArgs`)
```python
vocab_size: 50257          # Tokenizer vocabulary 
dim: 512                   # Hidden dimension
n_layers: 32               # Total decoder layers (1/4th are Mamba)
sliding_window: 2048       # SWA context size
num_experts: 8             # Total MoE experts per layer
num_experts_per_tok: 2     # Active experts per token (Top-K)
mamba_chunk_size: 64       # Hardware-aware chunking for selective scan
use_gradient_checkpointing: True # Enabled for memory-bound GPUs
```

## 4.3 Execution

The pipeline is completely self-contained. The `main()` execution block initializes the background data-loading threads, sets up Hessian-based metrics, and executes the active weight update pass.

### Running the Pretraining Pipeline

```bash
# Clone the repository
git clone https://github.com/yourusername/transmamba.git

# Navigate to the project directory
cd transmamba

# Ensure Kaggle/Hugging Face secrets are configured

# Execute the pretraining pipeline
python train.py
```

---

## 4.4 Resuming from Checkpoints

Checkpoints are designed to be hardware-agnostic and portable across different environments. During serialization, all `DataParallel` wrappers are removed to ensure seamless recovery regardless of the target hardware configuration.

Each checkpoint contains:

* Unwrapped model weights
* Stateful multi-driver optimizer matrices
* Exact byte offsets for the asynchronous data producer
* Training state required for deterministic continuation

### Resuming Training

To resume training from the latest checkpoint, enable the following setting in `TrainingConfig`:

```python
train_resume = True
```

The training pipeline will automatically locate and restore the checkpoint state, allowing training to continue from the last saved iteration.

---

## 📝 License

This project is open-sourced under the **MIT License**.
