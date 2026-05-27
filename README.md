# 🌾 Rice Leaf Disease Classification using SE-SPNet  
### Attention-Enhanced Lightweight Deep Learning Framework for Precision Agriculture

---

# 📌 Project Overview

This repository presents **SE-SPNet**, a custom lightweight convolutional neural network architecture designed for automated rice leaf disease classification using deep learning and attention-enhanced feature representation.

The framework is developed for:

- Precision Agriculture
- AI-assisted Crop Monitoring
- Smart Farming Systems
- Early Plant Disease Detection
- Explainable Agricultural AI

Unlike generic transfer learning pipelines, this project introduces a custom research-oriented CNN architecture integrating efficient spatial feature extraction and attention-guided learning.

---

# 🚀 Key Highlights

✅ Custom SE-SPNet Architecture  
✅ Attention-Enhanced Feature Learning  
✅ Lightweight Deep Learning Pipeline  
✅ Research-Oriented CNN Design  
✅ Explainable AI with Grad-CAM  

---

# 🧠 Proposed SE-SPNet Architecture

The proposed SE-SPNet framework combines:

- Convolutional feature extraction
- Attention-guided representation learning
- Lightweight spatial encoding
- Efficient disease discriminative learning
- Robust feature refinement mechanisms

The architecture is specifically optimized for:

- Computational efficiency
- Improved generalization
- Fine-grained disease localization
- Real-world agricultural deployment

---

# 🏗️ Model Architecture

<p align="center">
  <img src="results/model_architecture.png" width="900">
</p>

---

# 📂 Dataset

The dataset contains rice leaf images from multiple disease categories.

## Disease Classes

| Class ID | Disease Name |
|---|---|
| 0 | Bacterial Blight |
| 1 | Blast |
| 2 | Brown Spot |

---

# 🛠️ Technology Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Data Processing | NumPy / Pandas |
| Visualization | Matplotlib / Seaborn |
| Evaluation | Scikit-learn |
| Environment | Kaggle Notebook |

---

# 📁 Repository Structure

```text
Rice-Leaf-Disease-Classification-SE-SPNet/
│
├── notebooks/
│   └── se_spnet_rice_leaf_classification.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
│
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── gradcam_visualization.png
│   ├── gradcam_overlay.png
│   ├── model_architecture.png
│   └── sample_predictions.png
│
├── models/
│   └── trained_model.h5
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/Rice-Leaf-Disease-Classification-SE-SPNet.git
cd Rice-Leaf-Disease-Classification-SE-SPNet
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Training

```bash
python src/train.py
```

---

# 📊 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Specificity
- Confusion Matrix
- Grad-CAM Visualization

---

# 📈 Experimental Results

| Metric | Score |
|---|---|
| Accuracy | 99.24 |
| Precision | 99.26 |
| Recall | 99.24 |
| F1-Score | 99.24 |
| Specificity | 99.62 |

---

# 📉 Training Curves

## Accuracy Curve

<p align="center">
  <img src="results/accuracy_curve.png" width="700">
</p>

---

## Loss Curve

<p align="center">
  <img src="results/loss_curve.png" width="700">
</p>

---

# 🔍 Confusion Matrix

<p align="center">
  <img src="results/confusion_matrix.png" width="700">
</p>

---

# 🔬 Explainable AI using Grad-CAM

To improve interpretability and model transparency, Grad-CAM visualization is integrated into the pipeline.

The Grad-CAM module highlights disease-relevant spatial regions used by the network during classification.

## Grad-CAM Visualization

<p align="center">
  <img src="results/gradcam_visualization.png" width="900">
</p>

---

# 🌾 Applications

- Smart Agriculture
- Precision Farming
- AI-based Crop Monitoring
- Automated Plant Disease Diagnosis
- Edge-AI Agricultural Systems
- Agricultural Decision Support Systems

---

# 🔬 Research Contributions

### Key Contributions of this Work

- Proposed a custom lightweight CNN architecture (SE-SPNet)
- Integrated attention-enhanced feature learning
- Achieved strong disease classification performance
- Improved explainability using Grad-CAM
- Developed an industry-style reproducible deep learning workflow

---

# 📌 Future Work

Future extensions of this framework include:

- Real-time disease detection
- Mobile deployment
- Transformer-based attention integration
- Multi-crop disease classification
- Edge-device optimization
- Federated agricultural AI systems

---

# 📚 Publication

**SE_SPnet: Rice leaf disease prediction using stacked parallel convolutional neural network with squeeze-and-excitation**  
Parag Bhuyan et al.

📄 Paper: [Read Publication](https://doi.org/10.1111/exsy.13304)

If you use this work in your research, please cite.

---

# 👨‍💻 Author

## Dr. Parag Bhuyan

PhD Researcher — Artificial Intelligence & Computer Vision

### Research Interests

- Computer Vision
- Deep Learning
- Precision Agriculture
- Real Time Object Detection
- Medical & Agricultural Imaging

---

# ⭐ Acknowledgements

- TensorFlow
- Keras
- OpenCV
- Kaggle
- Open-source AI Research Community

---

# 📄 License

This project is licensed under the MIT License.

---

# 💻 Connect With Me

<p align="center">

<a href="https://github.com/parag-cv-ai">
  <img src="https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github">
</a>

<a href="https://www.linkedin.com/in/dr-parag-bhuyan/">
  <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin">
</a>

<a href="https://scholar.google.com/citations?user=X5zc0mEAAAAJ&hl=en/">
  <img src="https://img.shields.io/badge/Google%20Scholar-Research-red?style=for-the-badge&logo=google-scholar">
</a>

</p>

---

# 📌 Project Features

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10-blue">
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange">
<img src="https://img.shields.io/badge/Deep%20Learning-CNN-success">
<img src="https://img.shields.io/badge/Computer%20Vision-Agricultural%20AI-green">
<img src="https://img.shields.io/badge/Precision-Farming-brightgreen">
<img src="https://img.shields.io/badge/Custom%20Architecture-SE--SPNet-blue?style=flat-square">
<img src="https://img.shields.io/badge/Attention-Mechanism-success?style=flat-square">
<img src="https://img.shields.io/badge/Explainable-AI-success">
</p>

---

<p align="center">
  ⭐ If you found this project useful, consider giving it a star!
</p>
