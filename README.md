# 🌍 Tourism_Experience_Analytics_Platform

A production-grade **Machine Learning–powered tourism intelligence system** that predicts attraction ratings, classifies visit modes, and generates personalized attraction recommendations using model comparison, feature engineering, and automated best-model selection.

The system performs:
- Attraction Rating Prediction (Regression)
- Visit Mode Prediction (Classification)
- Personalized Attraction Recommendations
- Automatic Best Model Selection
- Manual Model Override
- Scaled Feature Pipeline
- Professional Streamlit UI with Glassmorphism
- Real-time prediction display
- System health monitoring

---
---

## 🚀 Key Highlights

- End-to-end ML pipeline (data → model → deployment)
- Multiple model comparison framework
- Automatic best-model selection using JSON config
- Manual model switching option
- Feature engineering & scaling pipeline
- Reverse scaling for clean rating outputs
- Confidence score display for classification
- Personalized recommendation engine
- Streamlit UI with glassmorphism design
- Safe model loading & fallback handling
- Production-ready modular structure

---
---

## 📁 Project Structure

```
Tourism_Experience_Analytics/
│
├── assets/
│   └── styles.css
│
├── data/
│   └── processed/
│       └── master_dataset.csv
│
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest_regression.pkl
│   ├── gradient_boosting_regression.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest_clf.pkl
│   ├── gradient_boosting_clf.pkl
│   ├── xgboost_clf.pkl
│   ├── lightgbm_clf.pkl
│   ├── scaler.pkl
│   └── label_encoders.pkl
│
├── reports/
│   ├── best_regression_model.json
│   ├── best_classification_model.json
│   ├── regression_model_comparison.csv
│   └── classification_model_comparison.csv
│
├── notebooks/
│   └── Tourism_Experience_Analytics.ipynb
│
├── aap.py
├── requirements.txt
└── README.md
```

---
---

## Download full project from google drive.

Link - https://drive.google.com/drive/folders/1xnFYtf1xGj2V-AXGb0YYis1AN2OyCbcH?usp=drive_link

## ⚙️ Installation & Setup (Step by Step)

### 1. Clone the Repository

```
git clone https://github.com/your-username/Tourism_Experience_Analytics.git
```

```
cd Tourism_Experience_Analytics
```

---

### 2. Create Virtual Environment

Creating:

For Windows

```
py -m venv venv
```


For Mac
```
python -m venv venv
```

or

```
python3 -m venv venv
```

Activate:

For macOS / Linux:
```
source venv/bin/activate
```

For Windows:
```
venv\Scripts\activate
```

---

### 3. Install Requirements

```
pip install -r requirements.txt
```

---

### 4. Run the Application

```
streamlit run app.py
```

---
---

## 🧠 System Architecture (High Level)

```
Raw Data
↓
Data Cleaning & Merging
↓
Feature Engineering
↓
Feature Scaling & Encoding
↓
Train Multiple Models
↓
Model Comparison
↓
Best Model Selection (JSON Config)
↓
Streamlit Deployment
```

The application dynamically loads models and selects the best-performing one automatically.

---
---

## 📊 Core Functional Modules

### 📈 1. Regression Module (Rating Prediction)

Predicts the expected rating (1–5 scale) for a tourist attraction.

Inputs:
- Continent
- Visit Year
- Visit Month

Pipeline:
- Automatic categorical encoding
- Feature alignment with trained model
- Scaling
- Prediction
- Reverse scaling to original rating scale
- Clamping between 1 and 5

Output:
- Predicted rating
- Model used
- Clean UI card display

---

### 🎯 2. Classification Module (Visit Mode Prediction)

Predicts visit category:

- Business
- Family
- Couples
- Friends
- Solo

Pipeline:
- Encoding
- Feature alignment
- Prediction
- Probability extraction (if supported)
- Confidence score display

Output:
- Predicted class
- Confidence percentage
- Model used

---

### ⭐ 3. Recommendation Engine

Generates personalized attraction suggestions based on:

- User history
- Attraction type preference
- Popularity signals

Logic:
- Identify user's favorite attraction type
- Rank attractions by popularity
- Return recommendations

Output:
- Attraction name
- Popularity score
- Glass-style card UI

---
---

## 🧠 Model Comparison Framework

### 🔹 Regression Models
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Best model selected using **R² Score**

---

### 🔹 Classification Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

Best model selected using **Accuracy**

---

### 📄 Best Model JSON Example

```
{
  "task": "regression",
  "model_name": "Gradient Boosting (Tuned)",
  "model_path": "./models/gradient_boosting_regression.pkl",
  "metric": "R2",
  "score": 0.7453
}
```

The app reads this file to automatically load the best-performing model.

---
---

## 🎨 Streamlit UI Features

### 📊 Tabs Layout
- Rating Prediction
- Visit Mode Prediction
- Recommendation Engine

### 🎛 Sidebar Controls
- Automatic (Best) Model Selection
- Manual Model Override
- System Health Status
- Dataset Load Status
- Scaler Status
- Encoder Status
- Best Model Status

### 🎨 Design System
- Glassmorphism UI
- Gradient background
- Blurred card design
- Clean metric display
- Responsive layout
- Unique widget keys to prevent duplication errors

### 📊 Prediction Display
- Styled glass cards
- Clear metric emphasis
- Confidence visualization (for classification)

---
---

## 🛡 System Stability Features

- Safe model loading
- JSON-based best model loading
- Feature name alignment protection
- Missing feature fallback
- Automatic encoding handling
- Reverse scaling for regression
- Range clamping for ratings
- Duplicate widget key protection
- Version-safe loading handling

---
---

## 📌 Ideal Use Cases

- Travel analytics platforms
- Tourism intelligence dashboards
- Personalization engines
- ML deployment portfolios
- End-to-end ML system demos
- SaaS-based tourism optimization tools

---
---

## 🚀 Future Scope & Enhancements

1️⃣ Hybrid recommendation system  
2️⃣ Deep learning–based prediction models  
3️⃣ Real-time API integration  
4️⃣ Seasonal trend analysis  
5️⃣ Multi-country tourism dashboard  
6️⃣ SHAP explainability  
7️⃣ Interactive maps  
8️⃣ User login & profile saving  
9️⃣ Auto retraining pipeline  
🔟 Cloud deployment  
1️⃣1️⃣ Regression confidence intervals  
1️⃣2️⃣ Data drift monitoring  

---
---

## 🏁 Conclusion

This project demonstrates the successful design and deployment of a production-grade Machine Learning platform for tourism analytics. It integrates structured feature engineering, model comparison logic, and automated model selection into a user-friendly Streamlit interface.

By combining regression, classification, and recommendation systems in one unified application, the platform showcases real-world ML deployment practices. The modular design ensures extensibility, maintainability, and scalability.

This system reflects strong applied ML engineering, production awareness, and clean UI integration — making it suitable for professional portfolios and real-world tourism intelligence applications.

---
---

## 🤝 Author

### **Predeep Kumar**

🧑‍💻 Machine Learning Engineer | Applied AI Systems | Production ML Deployment  

Built with ❤️ as a full-stack Machine Learning analytics system demonstrating real-world deployme
