<div align="center">
<h2>Power Plant Energy Predictor ⚡️</h2>
<h3>A Predictive Energy Output Intelligence App</h3>
<p>An interactive dashboard to forecast the energy output of a Combined Cycle Power Plant using machine learning, built with Streamlit, Scikit-learn, and TensorFlow.</p>
</div>

---

## 📋 Overview:
The **Power Plant Energy Predictor** is an end-to-end machine learning application that forecasts the net hourly electrical energy output of a power plant. It allows users to visualize key data trends from sensor readings, compare the performance of multiple predictive models, and get instant energy output predictions, all within a user-friendly web interface.

### 🎯 Why This Dashboard?
✅ **For Energy Analysts & Engineers**:  Explore how ambient conditions like temperature and pressure directly impact power generation in a real-world scenario.<br>
✅ **For Data Scientists & Developers**: See a practical, deployed example of a full machine learning workflow—from data analysis and model training to building an interactive Streamlit user interface.<br>

---

## ✨ Features:
✔️ **Interactive Data Visuals** – Explore the dataset with dynamic charts for variable distributions, a correlation heatmap, and a full pairplot to see relationships between features.<br>
✔️ **Multiple Prediction Models** – Compare the performance of several models: Linear Regression, Random Forest, XGBoost, and an Artificial Neural Network (ANN).<br>
✔️ **Live Energy Prediction** – Input the ambient temperature, exhaust vacuum, ambient pressure, and relative humidity to receive an instant prediction and a pie chart visualization.<br>
✔️ **Transparent Model Evaluation** – Review detailed tables showing actual vs. predicted values (and the difference) for each model on a hold-out test set.<br>
✔️ **Dynamic Performance Plots** – Instantly visualize each model's accuracy with "Actual vs. Predicted" and "Residuals Distribution" plots.

---

## 🚀 Live Demo:
🔗 Try the Power Plant Energy Predictor Now: https://power-plant-predictor-qs5j6u6lbg4jxh6cae5dkn.streamlit.app/


---

## 🔧 Installation & Setup:
### **Prerequisites:**
-Python 3.8+ <br>
-Git <br>

### **Local Setup:**
Clone the repository:
```bash
git clone https://github.com/Neelanjan2448040/Power-Plant-Predictor.git
cd Power-Plant-Predictor
```

Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate   # For Windows
# OR
source venv/bin/activate  # For macOS/Linux
```

Install the required packages:
```bash
pip install -r requirements.txt
```

Run the Streamlit application:
```bash
streamlit run app.py
```
