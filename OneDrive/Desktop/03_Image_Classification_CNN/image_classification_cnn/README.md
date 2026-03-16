# 🖼️ Image Classification using CNN

A Deep Learning project that trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset to classify images into 10 categories, with a Streamlit web app for live predictions.

---

## 📁 Project Structure

```
image_classification_cnn/
├── app.py              # Streamlit web app
├── train.py            # CNN training script
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the CNN
```bash
python train.py
```
Downloads CIFAR-10 automatically, trains the CNN, saves `cnn_model.h5` and `training_curves.png`.

### 3. Launch the app
```bash
streamlit run app.py
```

---

## 🏷️ Categories (CIFAR-10)

Airplane · Automobile · Bird · Cat · Deer · Dog · Frog · Horse · Ship · Truck

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | CNN model |
| OpenCV / Pillow | Image preprocessing |
| Matplotlib | Training curves |
| Streamlit | Web interface |

---

## 🧠 CNN Architecture

- **3 Convolutional Blocks** (32 → 64 → 128 filters)
- **BatchNormalization** after each conv layer
- **MaxPooling + Dropout** to prevent overfitting
- **Dense(256) → Softmax(10)** classifier
- **Data Augmentation**: rotation, flips, shifts, zoom
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

*Built by Muhammed Anshad M*
