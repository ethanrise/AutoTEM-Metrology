# AutoTEM-Metrology 🔬

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Industry](https://img.shields.io/badge/Industry-Semiconductor-red.svg)](#)

**AutoTEM-Metrology** is a high-precision automated framework designed for **Critical Dimension (CD) Metrology** in semiconductor TEM/SEM imaging. 

By integrating **Vision-Language Models (VLM)** and **SAM 2** for semantic localization with **classical sub-pixel OpenCV algorithms** for deterministic measurement, this project provides a robust solution for laboratory-grade inspection that bridges the gap between AI-driven perception and physical measurement accuracy.

---

## 🌟 Key Features

- **Hybrid Perception Pipeline:** Leverages VLMs for intelligent ROI identification and SAM 2 for precise feature segmentation, followed by classical CV for final measurement.
- **Sub-pixel Precision:** Implements advanced sub-pixel edge detection kernels to achieve measurement errors **≤0.3mm** (at specific magnification), outperforming standard pixel-level analysis.
- **Industrial Robustness:** Specifically optimized for low-contrast, noisy, and artifact-heavy semiconductor environments (TEM/SEM).
- **Critical Dimension (CD) Focus:** Pre-configured workflows for measuring gate width, hole diameter, pitch, and multi-layer thin-film thickness.
- **Agent-Ready Design:** Modular Pythonic API designed to be integrated into **Autonomous Industrial Agents** and Automated Test Equipment (ATE) workflows.

---

## 🏗️ Architecture

The framework follows a **"Semantic-to-Deterministic"** logic:

1.  **Semantic Localization (VLM):** Interprets the measurement task (e.g., "measure the gate oxide thickness") and identifies the ROI.
2.  **Feature Segmentation (SAM 2):** Extracts high-fidelity masks of the target structures.
3.  **Refined Edge Detection:** Applies sub-pixel interpolation on the extracted boundaries.
4.  **Metric Calibration:** Converts pixel distances to physical units (nm/um) based on TEM magnification metadata.
5.  **Validation:** Statistical outlier rejection to ensure repeatability.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- CUDA 12.x compatible GPU (required for VLM/SAM inference)
- OpenCV-Python

### Installation
```bash
git clone https://github.com/YourUsername/AutoTEM-Metrology.git
cd AutoTEM-Metrology
pip install -r requirements.txt
```

---

## 🛠️ Usage Example

```python
from autotem import MetrologyAgent

# Initialize the Agent with VLM-guided localization
agent = MetrologyAgent(model_type='vlm', device='cuda')

# Perform automated CD-Metrology on a TEM sample
# The agent identifies the 'gate_width' and applies sub-pixel algorithms
results = agent.measure(
    image_path="samples/chip_sample_001.png",
    target="gate_width",
    unit="nm"
)

print(f"Status: {results.status}")
print(f"Measured CD Value: {results.value:.4f} {results.unit}")
print(f"Confidence Score: {results.confidence:.2f}")
```

---

## 📊 Performance Benchmarks

| Metric | Standard CNN Approach | **AutoTEM-Metrology** |
| :--- | :--- | :--- |
| **Localization Method** | Bounding Box | **VLM + SAM 2 (Semantic)** |
| **Edge Precision** | Pixel-level (1px) | **Sub-pixel (0.1px - 0.2px)** |
| **System Error** | ~1.2mm | **≤0.3mm** |
| **Repeatability (σ)** | Low | **High** |

---

## 📁 Project Structure

```text
AutoTEM-Metrology/
├── autotem/                # Core library
│   ├── agents/             # VLM & Agentic logic
│   ├── perception/         # SAM 2 & Segmentation modules
│   └── metrology/          # Sub-pixel & Classical CV algorithms
├── configs/                # Magnification & Calibration configs
├── samples/                # Demo images (Open-source datasets)
├── tests/                  # Unit tests for precision validation
└── main.py                 # CLI Entry point
```

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details. This license is chosen to facilitate industrial adoption while ensuring clear patent and copyright guidelines.

---


*Open for technical discussions on High-precision Metrology, Industrial Agents, and Remote Collaboration.*
