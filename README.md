# Robust Spam SMS Detection with Real and Synthetic Data

## Overview

This project develops a robust SMS spam detection system by combining real-world SMS data with synthetically generated obfuscated spam messages. The approach leverages multiple deep learning architectures and multi-loss training strategies to improve model robustness against adversarial text transformations.

### Key Innovation
We test classifier robustness by generating 11 levels of increasingly obfuscated spam messages, allowing us to evaluate how well models generalize to evolving spam evasion techniques.

---

## Project Pipeline

```
1. Real SMS Data (UCI Dataset)
                    ↓
2. Synthetic Data Generation (11 obfuscation levels via Qwen3B)
                    ↓
3. Heuristic Filtering (Level validation & deduplication)
                    ↓
4. Semantic Filtering (Recovery scoring & quality control)
                    ↓
5. Train/Validation/Test Split
                    ↓
6. Model Training (DistilBERT, FastText, TextCNN, CharCNN)
                    ↓
7. Evaluation (F1, Precision, Recall on Real & Synthetic subsets)
```

---

## Data Construction

### Real Data
- Source: UCI SMS Spam Collection Dataset
- Size: ~5,574 messages (ham and spam)
- Format: Label-Message pairs

### Synthetic Data Generation

The system generates 11 obfuscation levels for each spam message:

| Level | Technique | Example |
|-------|-----------|---------|
| L1 | Simple number replacement | S1X, C4SH, Chan3 |
| L2 | Special character substitution | S!X, CA$H, C@SH |
| L3 | Abbreviations and slang | txt, gr8, & |
| L4 | Delimiter insertion | S.I.X, C-A-S-H |
| L5 | Word spacing | s p a c e d t e x t |
| L6 | Homoglyph substitution | Greek letters substituted |
| L7 | Phonetic + symbol mix | K-A-S-H, 3nd 2 |
| L8 | Phone number disruption | 8-7-5-7-5 |
| L9 | Unicode block characters | Unicode variations |
| L10 | Extreme structures | [S].[I].[X], {{C}}{{A}}{{S}}{{H}} |
| L11 | Mixed combination | 3+ techniques combined |

Model Used: Qwen3-32B quantized (Q4_K_M GGUF format)

### Filtering Stages

Stage 1: Heuristic Filtering
- Remove exact duplicates
- Validate obfuscation level characteristics
- Preserve semantic meaning

Stage 2: Semantic Filtering
- Recovery Score: Similarity between original and recovered message
- Threshold: recover_score > 0.2
- Uses character-level recovery and fuzzy string matching

---

## Models

### DistilBERT
- Transformer-based pre-trained model
- Input: Tokenized text (max 64 tokens)
- Output: Binary spam/ham classification
- Pros: Best overall performance, excellent generalization
- Cons: Slower inference, higher memory usage

### FastText
- Fast linear text classifier
- Input: Character n-grams
- Output: Binary classification
- Pros: Extremely fast, resource-efficient
- Cons: Lower accuracy on complex patterns

### TextCNN
- 1D Convolutional Neural Network
- Multiple filter sizes: [2, 3, 4, 5]
- Output: Binary classification
- Pros: Good balance of speed and accuracy

### CharCNN
- Character-level CNN (2-layer deep)
- Input: Character indices (max 128 chars)
- Pros: Best obfuscation resistance with synthetic data
- Cons: Fails on real-only data without synthetic training

---

## Training Configurations

### Model 1 (M1): Real Data Only
Training: Real SMS only
Purpose: Baseline performance

### Model 2 (M2): Real + Synthetic Combined
Training: Real + Synthetic messages
Purpose: Test robustness to obfuscation

### Model 3 (M3): Entropy Loss
Loss = Classification Loss + 0.1 * Entropy Loss
Purpose: Encourage diverse decision boundaries

### Model 4 (M4): Multi-Loss Training
Loss = L_cls + alpha*L_rob + beta*L_recon + gamma*L_group + eta*L_entropy

---

## Training Details

### Hyperparameters
- Optimizer: AdamW with learning rate scheduling
- Learning Rate: 5e-5 (with linear warmup over 100 steps)
- Batch Size: 32
- Epochs: 10 (with early stopping)
- Gradient Clipping: 1.0
- Dropout: 0.1-0.5

### Advanced Techniques
- Mixed Precision Training (AMP): Reduces memory
- Multi-GPU Support: DataParallel distribution
- Early Stopping: Patience of 3 epochs on validation F1
- Learning Rate Scheduler: Linear warmup + decay

### Data Split
- Training: 70%
- Validation: 10%
- Test: 20%

---

## Evaluation Metrics

### Primary Metrics
- Precision: True Positives / (TP + FP)
- Recall: True Positives / (TP + FN)
- F1-Score: Harmonic mean of Precision and Recall

### Evaluation Subsets
Each model evaluated on three subsets:
1. Full Test Set: Combined real and synthetic samples
2. Real-Only Subset: Only authentic SMS messages
3. Synthetic-Only Subset: Only obfuscated variations

---

## Key Findings

- DistilBERT: Best overall choice for production
- FastText: Ideal for real-time and edge deployments
- TextCNN: Strong balance between accuracy and efficiency
- CharCNN: Superior obfuscation handling but requires synthetic training

### Critical Observation
CharCNN fails catastrophically when trained on real data only (F1 drops to 0.2). This suggests character patterns alone are insufficient without synthetic training signal.

---

## Output Files

### Models
- models/Model1_best.pt - Real-only baseline
- models/Model2_best.pt - Real + Synthetic
- models/Model3_best.pt - Entropy regularized
- models/Model4_best.pt - Multi-loss CharCNN

### Metrics
- final_metrics.csv - Complete F1/Precision/Recall breakdown
- df_stage2_out.csv - Filtered synthetic dataset with recovery scores
- loss_curves.png - Training loss visualization

---

## Future Improvements

- Contrastive learning for similar message clustering
- Hybrid architectures combining multiple models
- Adversarial training for robust optimization
- Domain adaptation for regional/language-specific datasets
- Model distillation for lightweight deployment

---

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
tqdm>=4.65.0
rapidfuzz>=3.0.0
llama-cpp-python>=0.2.0
```

Installation:
```bash
pip install torch transformers pandas numpy scikit-learn tqdm rapidfuzz
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

---

## Usage

1. Generate synthetic data: python generate.ipynb
2. Filter and score data: python filtering.ipynb
3. Train all models: python training.ipynb

---

## Project Structure

```
Learning/SM/
├── generate.ipynb
├── training.ipynb
├── heuristic +semantic filtering.ipynb
├── README.md
├── models/
├── final_metrics.csv
└── data/
```
