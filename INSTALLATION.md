# Installation Guide

## Quick Start

### Option 1: Minimal Installation (Recommended for Testing)
Install only essential dependencies:

```bash
pip install -r requirements-minimal.txt
```

### Option 2: Full Installation (Recommended for Development)
Install all dependencies including visualization and development tools:

```bash
pip install -r requirements.txt
```

### Option 3: GPU/CUDA Support (Recommended for Training)
If you have NVIDIA GPU with CUDA toolkit installed:

```bash
pip install -r requirements-cuda.txt
```

Then install llama-cpp-python with CUDA support:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

---

## Python Version Requirements

- Python 3.8 or higher
- Recommended: Python 3.10+

Check your Python version:
```bash
python --version
```

---

## System Requirements

### Minimum Requirements
- 8GB RAM
- 10GB disk space

### Recommended Requirements
- 16GB+ RAM
- 50GB disk space
- NVIDIA GPU with 6GB+ VRAM (for training)

---

## Installation Steps

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies

Choose one of the requirements files above based on your needs.

### 4. Verify Installation

```python
python -c "
import torch
import transformers
import pandas
import numpy
print('All core packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

---

## Troubleshooting

### Issue: CUDA not detected

If you have NVIDIA GPU but CUDA is not detected:

1. Install NVIDIA drivers
2. Install CUDA toolkit
3. Reinstall PyTorch:
   ```bash
   pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu118
   ```

### Issue: llama-cpp-python installation fails

For CUDA support:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --no-cache-dir llama-cpp-python
```

For CPU only:
```bash
pip install llama-cpp-python
```

### Issue: Transformers model downloading

First time running will download models (~500MB each). Ensure stable internet connection.

---

## Package Descriptions

| Package | Purpose | Version |
|---------|---------|---------|
| torch | Deep learning framework | >=2.0.0 |
| transformers | Pre-trained NLP models | >=4.30.0 |
| numpy | Numerical computing | >=1.24.0 |
| pandas | Data manipulation | >=1.5.0 |
| scikit-learn | Machine learning utilities | >=1.2.0 |
| tqdm | Progress bars | >=4.65.0 |
| rapidfuzz | String similarity | >=3.0.0 |
| matplotlib | Visualization | >=3.7.0 |
| jupyter | Interactive notebooks | >=1.0.0 |
| llama-cpp-python | GGUF model inference | >=0.2.0 |

---

## Virtual Environment Management

### Save current environment state
```bash
pip freeze > current-requirements.txt
```

### Recreate environment from file
```bash
pip install -r current-requirements.txt
```

### Updating Dependencies

To update all packages to latest compatible versions:

```bash
pip install -r requirements.txt --upgrade
```

To update specific package:

```bash
pip install --upgrade package_name
```

---

## Next Steps

After installation, you can:

1. **Generate Synthetic Data**: Run `generate.ipynb`
2. **Filter Data**: Check available notebooks
3. **Train Models**: Run training notebooks
4. **View Documentation**: See `README.md`

---

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review official package documentation
3. Check PyTorch forums: https://discuss.pytorch.org/
4. Check Hugging Face forums: https://discuss.huggingface.co/

---

## Uninstallation

To remove the virtual environment:

```bash
# Deactivate first
deactivate

# Remove virtual environment
rm -rf venv  # On Windows: rmdir /s venv
```
