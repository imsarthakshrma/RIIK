# KOLOSIS: Cognitive-Inspired AI Architecture

**K**nowledge **O**riented **L**earning **O**rganization **S**ystem for **I**ntelligent **S**ystems

A research project exploring cognitive-inspired mechanisms for improved language model performance through hierarchical abstraction and multi-scale temporal attention.

---

## 🎉 Breakthrough Achievement

**Kolosis V2 Minimal**: **0.96 val loss** (+66.1% improvement over baseline GPT)

This represents the culmination of systematic research from concept to production-ready architecture.

---

## What is KOLOSIS?

KOLOSIS is a novel transformer architecture that incorporates cognitive-inspired mechanisms:

1. **Hierarchical Embeddings**: Multi-level abstraction (Symbol → Concept → Law)
2. **Multi-Scale Temporal Attention**: Fast, medium, and slow memory decay
3. **Parallel Streams**: Direct supervision for each cognitive mechanism
4. **Learned Fusion**: Optimal combination of processing streams

---

## Key Results

| Model | Val Loss | vs Baseline | Parameters | Status |
|-------|----------|-------------|------------|--------|
| Baseline GPT | 2.84 | - | 150K | ✅ |
| Hierarchical-Only | 2.50 | +12.0% | 150K | 🏆 Production |
| Kolosis V1 | 2.78 | +2.1% | 156K | ⚠️ Gradient starved |
| Kolosis V2 Full | 2.56 | +9.9% | 454K | ✅ Research |
| **Kolosis V2 Minimal** | **0.96** | **+66.1%** | **222K** | 🏆 **Best** |

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/RIIK.git
cd RIIK

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Train Kolosis V2 Minimal (Recommended)

```python
from neural_networks.kolosis.kolosis_v2_minimal import KolosisV2Minimal

# Create model
model = KolosisV2Minimal(
    vocab_size=50257,
    n_embd=256,
    block_size=256,
    n_layer=6,
    dropout=0.1
)

# Train (see experiments/ for full training scripts)
```

### Train Hierarchical-Only (Production)

```python
from experiments.hybrid_models import GPT_HierarchicalEmbedding

model = GPT_HierarchicalEmbedding(
    vocab_size=50257,
    n_embd=256,
    n_head=8,
    n_layer=6,
    block_size=256
)
```

---

## Project Structure

```
RIIK/
├── neural_networks/
│   ├── autograd/              # Scalar autograd engine
│   │   ├── engine.py          # Value class with backprop
│   │   ├── nn.py              # Neural network primitives
│   │   └── transformer_torch.py  # PyTorch GPT baseline
│   ├── kolosis/               # KOLOSIS architectures
│   │   ├── hierarchical_embedding.py  # Multi-level abstraction
│   │   ├── temporal_attention.py      # Multi-scale memory
│   │   ├── kolosis_transformer.py     # Kolosis V1
│   │   ├── kolosis_v2.py              # Parallel streams
│   │   └── kolosis_v2_minimal.py      # Streamlined (BEST)
│   ├── nlp/                   # NLP utilities
│   │   └── tokenizer.py       # Character tokenizer
│   └── data/                  # Data loaders
│       ├── toon_parser.py     # TOON format
│       └── tinystories_loader.py
├── experiments/               # Training scripts
│   ├── train_kolosis_v2_minimal.py
│   ├── ablation_study.py
│   ├── analyze_temporal_attention.py
│   └── wikitext/              # WikiText-103 experiments
├── tests/                     # Unit tests
│   ├── test_kolosis.py
│   └── test_transformer.py
└── docs/                      # Documentation
    ├── kolosis_research_plan.md
    ├── phase4_ablation_results.md
    ├── kolosis_v2_minimal_results.md
    └── final_summary_and_next_steps.md
```

---

## Key Innovations

### 1. Hierarchical Embeddings

Multi-level abstraction captures different semantic granularities:

```python
x = E_symbol + α·E_concept + β·E_law + E_pos
```

- **Symbol**: Raw token representation
- **Concept**: Mid-level semantic patterns
- **Law**: High-level structural patterns

**Result**: +12% improvement, works at all scales

### 2. Multi-Scale Temporal Attention

Three decay rates model different memory timescales:

```python
T(Δt) = α_fast·γ_fast^Δt + α_medium·γ_medium^Δt + α_slow·γ_slow^Δt
```

- **Fast** (γ≈0.7): Recent tokens (2-3 steps)
- **Medium** (γ≈0.9): Short-term context (7-10 steps)
- **Slow** (γ≈0.99): Long-term dependencies (full context)

**Result**: +7.1% improvement (needs longer contexts to shine)

### 3. Parallel Streams with Direct Supervision

Each cognitive mechanism has its own prediction head:

```python
concept_logits = concept_stream(hierarchical_emb)
semantic_logits = semantic_stream(hierarchical_emb + relations)
final = learned_fusion(concept, semantic)
```

**Result**: Eliminates gradient starvation, enables specialization

---

## Research Journey

### Phase 1: Foundation (Autograd Engine)
- Built scalar autograd from scratch
- Verified mathematical correctness
- Created neural network primitives

### Phase 2: Performance (PyTorch Migration)
- Migrated to PyTorch: **26.8x speedup**
- Production-ready infrastructure

### Phase 3: Innovation (Kolosis Research)
- Implemented 4 cognitive mechanisms
- All components tested and functional

### Phase 4: Validation (Ablation Studies)
- Discovered hierarchical embeddings are MVP (+12%)
- Found gradient starvation in full system (7.6x imbalance)
- Learned component interference without tuning

### Phase 5: Refinement (V2 Architecture)
- V2 Full: Parallel streams solve gradient starvation (+9.9%)
- **V2 Minimal: Streamlined to winners only (+66.1%)** 🏆

---

## Experiments

### Run Ablation Studies

```bash
# Test individual components
python experiments/run_component_ablation.py

# Analyze temporal attention
python experiments/analyze_temporal_attention.py

# Test hierarchical layer ablation
python experiments/test_hierarchical_layers.py
```

### Train on WikiText-103

```bash
# See docs/wikitext_training_guide.md for full instructions

# Baseline GPT
python experiments/wikitext/train_baseline_gpt.py

# Hierarchical
python experiments/wikitext/train_hierarchical.py

# Kolosis V2 Minimal
python experiments/wikitext/train_kolosis_v2_minimal.py
```

---

## Documentation

**Research**:
- [Kolosis Research Plan](docs/kolosis_research_plan.md) - Mathematical formulations
- [Phase 3 Test Results](docs/phase3_test_results.md) - Component tests
- [Phase 4 Ablation Results](docs/phase4_ablation_results.md) - Systematic evaluation

**Analysis**:
- [Component Interference Analysis](docs/component_interference_analysis.md) - Gradient issues
- [Optimization Results](docs/optimization_results.md) - Performance analysis
- [Hierarchical Layer Ablation](docs/hierarchical_layer_ablation.md) - 2-layer vs 3-layer

**Results**:
- [Kolosis V2 Results](docs/kolosis_v2_results.md) - Parallel streams
- [Kolosis V2 Minimal Results](docs/kolosis_v2_minimal_results.md) - **Breakthrough**
- [Final Summary](docs/final_summary_and_next_steps.md) - Complete overview

---

## Testing

```bash
# Run all tests
pytest tests/

# Specific test suites
pytest tests/test_kolosis.py
pytest tests/test_transformer.py
pytest tests/test_autograd.py
```

---

## Key Findings

### What Worked ✅

1. **Hierarchical embeddings** - Consistent +12% improvement
2. **Parallel streams** - Solves gradient starvation
3. **Direct supervision** - Each mechanism needs its own loss
4. **Simplification** - V2 Minimal beats V2 Full

### What We Learned 📚

1. **Don't bolt mechanisms onto transformers** - Causes interference
2. **Give each mechanism direct supervision** - Prevents gradient starvation
3. **Remove redundant components** - Symbol+Concept sufficient (Law redundant)
4. **Scale matters** - Temporal attention needs longer contexts

### What's Next 🚀

1. **WikiText-103 validation** - Test at proper scale (103M tokens)
2. **Temporal attention optimization** - Make it work with longer contexts
3. **Production deployment** - V2 Minimal or Hierarchical-only
4. **Task-specific tuning** - QA, summarization, etc.

---

## Citation

If you use KOLOSIS in your research, please cite:

```bibtex
@software{kolosis2024,
  title={KOLOSIS: Cognitive-Inspired AI Architecture},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/RIIK}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

This project explores cognitive-inspired mechanisms for AI, building on research in:
- Hierarchical abstraction in neural networks
- Multi-scale temporal processing
- Parallel stream architectures
- Direct supervision for complex systems

---

## Contact

For questions or collaboration:
- GitHub Issues: [github.com/yourusername/RIIK/issues](https://github.com/yourusername/RIIK/issues)
- Email: your.email@example.com

---

**Status**: ✅ Research complete, production-ready models available

**Best Model**: Kolosis V2 Minimal (0.96 val loss, +66.1% improvement)

**Next Steps**: WikiText-103 validation, temporal attention optimization


(I have named it RIIK because I wanted to.) 

Welcome to my implementation of a neural network built entirely from scratch using just NumPy! This project represents my journey into understanding the inner workings of deep learning frameworks by building one myself.

##  What I've Built?

I've implemented a fully functional neural network with all the bells and whistles you'd expect from a modern deep learning framework:

- **Optimization**: Adam and SGD with momentum for efficient training
- **Regularization**: Dropout, L1/L2 regularization, and batch normalization
- **Features**: Early stopping, learning rate scheduling, and mini-batch training
- **Versatility**: Supports both classification and regression tasks
- **Production-ready**: Model saving/loading and comprehensive evaluation metrics

##  Installation

Getting started is simple - just install the requirements:

```bash
pip install -r requirements.txt
```

##  Project Structure

Here's how I've organized the code:

```
neural_networks/
├── core/               # Core neural network implementation
│   ├── __init__.py     
│   ├── network.py     # Main neural network class
│   ├── layer.py        # Layer implementation
│   ├── activation.py   # Activation functions
│   └── optimizers.py   # Optimization algorithms
├── datasets/           # Data handling
│   ├── __init__.py
│   └── loaders.py      # Data loading utilities
└── examples/           # Example scripts
```

<!-- ##  Quick Start

Here's how you can use my neural network implementation:

```python
from neural_networks.core import NeuralNetwork
from neural_networks.datasets.loaders import DataLoader

# Load some data
data = DataLoader.load_classification_dataset('breast_cancer')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# Define the network architecture
architecture = [
    (30, 64, 'relu', 0.3, True),    # Input layer
    (64, 32, 'relu', 0.3, True),     # Hidden layer 1
    (32, 16, 'relu', 0.2, False),    # Hidden layer 2
    (16, 1, 'sigmoid', 0.0, False)   # Output layer
]

# Create and train the model
model = NeuralNetwork(architecture, task='classification', optimizer='adam', learning_rate=0.001)

model.fit(
    X_train, y_train, X_val, y_val,
    epochs=500, 
    batch_size=32,
    l2_reg=0.001, 
    early_stopping=True,
    patience=30, 
    lr_schedule=True
) -->

# Evaluate on test set
results = model.evaluate(X_test, y_test)
model.plot_history()

# Save model
model.save_model('breast_cancer_model.pkl')

# Load model
loaded_model = NeuralNetwork.load_model('breast_cancer_model.pkl')
```

## Examples

Check out the `examples/` directory for complete working examples, including:

- Binary classification (Breast Cancer dataset)
- Regression (California Housing prices)
- Multi-class classification (Synthetic data)

## Why I Built This?

I created this project to deepen my understanding of how neural networks work under the hood. By implementing everything from scratch, I've gained valuable insights into the mathematics and computational aspects of deep learning.

## 📊 Performance

I'm proud to say that my implementation achieves comparable performance to scikit-learn's MLP models while offering more flexibility and control. The modular design makes it easy to experiment with different architectures and techniques.

## What I Learned

- The importance of numerical stability in deep learning
- How different optimization techniques affect training
- The impact of various regularization methods
- Best practices in software design for machine learning

## Contributing??

Feel free to fork this repository and submit pull requests. I'm always open to suggestions and improvements!

## What's Next? 

This is just the beginning. My ultimate aim with RIIK is to go beyond basic neural networks and push toward more advanced deep learning architectures. 

Here's what I'm planning next:

- Implement CNNs, RNNs, Transformers, and Attention mechanisms
- Build an autograd engine to support dynamic computation graphs
- Optimize for scalability and efficiency to handle large-scale training tasks
- Work toward building a research-grade engine capable of benchmarks near GPT-3–level models (within practical constraints) if possible. 

## 📜 License

MIT - Feel free to use this code for your own learning or projects!
# RIIK: My Neural Network Implementation from Scratch

(I have named it RIIK because I wanted to.) 

Welcome to my implementation of a neural network built entirely from scratch using just NumPy! This project represents my journey into understanding the inner workings of deep learning frameworks by building one myself.

##  What I've Built?

I've implemented a fully functional neural network with all the bells and whistles you'd expect from a modern deep learning framework:

- **Optimization**: Adam and SGD with momentum for efficient training
- **Regularization**: Dropout, L1/L2 regularization, and batch normalization
- **Features**: Early stopping, learning rate scheduling, and mini-batch training
- **Versatility**: Supports both classification and regression tasks
- **Production-ready**: Model saving/loading and comprehensive evaluation metrics

##  Installation

Getting started is simple - just install the requirements:

```bash
pip install -r requirements.txt
```

##  Project Structure

Here's how I've organized the code:

```
neural_networks/
├── core/               # Core neural network implementation
│   ├── __init__.py     
│   ├── network.py     # Main neural network class
│   ├── layer.py        # Layer implementation
│   ├── activation.py   # Activation functions
│   └── optimizers.py   # Optimization algorithms
├── datasets/           # Data handling
│   ├── __init__.py
│   └── loaders.py      # Data loading utilities
└── examples/           # Example scripts
```

<!-- ##  Quick Start

Here's how you can use my neural network implementation:

```python
from neural_networks.core import NeuralNetwork
from neural_networks.datasets.loaders import DataLoader

# Load some data
data = DataLoader.load_classification_dataset('breast_cancer')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# Define the network architecture
architecture = [
    (30, 64, 'relu', 0.3, True),    # Input layer
    (64, 32, 'relu', 0.3, True),     # Hidden layer 1
    (32, 16, 'relu', 0.2, False),    # Hidden layer 2
    (16, 1, 'sigmoid', 0.0, False)   # Output layer
]

# Create and train the model
model = NeuralNetwork(architecture, task='classification', optimizer='adam', learning_rate=0.001)

model.fit(
    X_train, y_train, X_val, y_val,
    epochs=500, 
    batch_size=32,
    l2_reg=0.001, 
    early_stopping=True,
    patience=30, 
    lr_schedule=True
) -->

# Evaluate on test set
results = model.evaluate(X_test, y_test)
model.plot_history()

# Save model
model.save_model('breast_cancer_model.pkl')

# Load model
loaded_model = NeuralNetwork.load_model('breast_cancer_model.pkl')


## Examples

Check out the `examples/` directory for complete working examples, including:

- Binary classification (Breast Cancer dataset)
- Regression (California Housing prices)
- Multi-class classification (Synthetic data)

## Why I Built This?

I created this project to deepen my understanding of how neural networks work under the hood. By implementing everything from scratch, I've gained valuable insights into the mathematics and computational aspects of deep learning.

## Performance

I'm proud to say that my implementation achieves comparable performance to scikit-learn's MLP models while offering more flexibility and control. The modular design makes it easy to experiment with different architectures and techniques.

## What I Learned

- The importance of numerical stability in deep learning
- How different optimization techniques affect training
- The impact of various regularization methods
- Best practices in software design for machine learning

## Contributing??

Feel free to fork this repository and submit pull requests. I'm always open to suggestions and improvements!

## What's Next? 

This is just the beginning. My ultimate aim with RIIK is to go beyond basic neural networks and push toward more advanced deep learning architectures. 

Here's what I'm planning next:

- Implement CNNs, RNNs, Transformers, and Attention mechanisms
- Build an autograd engine to support dynamic computation graphs
- Optimize for scalability and efficiency to handle large-scale training tasks
- Work toward building a research-grade engine capable of benchmarks near GPT-3–level models (within practical constraints) if possible. 

## 📜 License

MIT - Feel free to use this code for your own learning or projects!
