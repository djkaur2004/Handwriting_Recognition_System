# ✍️ Handwriting Recognition System

An **end-to-end Deep Learning–based Handwriting Recognition System** built using **PyTorch**.  
This project converts handwritten text images into readable digital text using a **CRNN (CNN + BiLSTM)** architecture trained with **CTC Loss**, enabling alignment-free sequence recognition.

The project demonstrates a complete OCR pipeline, from data preprocessing and model training to evaluation and prediction.

---

## 🚀 Highlights
- Recognizes handwritten text directly from image inputs  
- Uses a **CRNN architecture (CNN + BiLSTM)** for combined visual and sequential learning  
- Employs **CTC Loss** to handle variable-length text without manual alignment  
- Supports **CUDA acceleration** for faster training on compatible GPUs  
- Generates performance graphs and evaluation metrics  
- Automatically saves the best-performing model based on validation loss  

---

## 🏗 Model Overview
- **CNN Backbone**  
  Extracts spatial and visual features from grayscale handwriting images  

- **BiLSTM (2 layers)**  
  Captures the sequential nature and context of handwritten text  

- **CTC Decoder**  
  Converts model outputs into readable text by removing blank and repeated characters  

---

## 🛠 Tech Stack
- Python  
- PyTorch, Torchvision  
- NumPy, PIL  
- Matplotlib, Scikit-learn  
- CUDA (GPU acceleration, if available)

---

## 📊 Dataset
This project is trained on a publicly available handwriting dataset.  
Due to licensing and size constraints, the dataset is not included in this repository.  
Please refer to the original dataset source for access and usage guidelines.

---

## ⚙️ Training & Evaluation
- Trained for **30 epochs** using the **Adam optimizer**  
- **Best model checkpoint** saved based on validation loss  
- Evaluation metrics include:
  - Precision  
  - Recall  
  - F1-Score  
  - Exact Match Accuracy  
- Training and validation loss curves are generated for performance analysis  

---

## 📁 Outputs
- Best trained model saved as `.pth`  
- Training and validation performance graphs  
- Sample prediction images with predicted vs ground truth text  
- All outputs stored in the `results/` directory  

---

## 🎯 Key Learnings
- Practical understanding of **CRNN architecture and CTC Loss**  
- Designing an OCR pipeline using **PyTorch**  
- Model training optimization and evaluation techniques  
- Building an end-to-end machine learning project  

---

## 🔮 Future Scope
- Multi-language handwriting recognition  
- Improved handling of cursive and complex handwriting styles  
- Deployment as a web or mobile application  

---

⭐ *If you find this project useful, feel free to star the repository.*
