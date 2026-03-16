# 🏠 House Price Prediction

A Machine Learning web app that predicts house prices based on property features using multiple regression models. Built with Python, Scikit-learn, and Streamlit.

---

## 📁 Project Structure

```
house_price_prediction/
├── app.py           # Streamlit web app
├── train.py         # Model training & comparison script
├── requirements.txt # Dependencies
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```
This compares Linear Regression, Ridge, Random Forest, and Gradient Boosting — then saves the best as `model.pkl`.

### 3. Launch the app
```bash
streamlit run app.py
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | ML models + preprocessing |
| Pandas / NumPy | Data manipulation |
| Matplotlib | Visualisation |
| Streamlit | Web interface |

---

## 💡 Features

- Input: area, bedrooms, bathrooms, age, floors, garage, garden, location, condition
- Compares 4 regression algorithms and picks best by R² score
- Real-time price prediction via interactive web UI

---

## 📌 Notes

- Replace the synthetic data in `train.py` with a real dataset for production use.
- Recommended dataset: [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

---

*Built by Muhammed Anshad M*
