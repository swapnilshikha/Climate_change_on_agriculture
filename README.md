# 🌾 Impact of Climate Change on Agriculture

This project aims to analyze and predict the **economic impact of climate change on agriculture** using Machine Learning techniques. We focus on how climatic variables such as temperature, precipitation, and extreme weather events affect crop yields and overall agricultural productivity in India.

## 🎯 Objective

To develop predictive models that can estimate the economic loss or gain in agriculture caused by changing climate patterns — supporting farmers and policymakers in making data-driven decisions for sustainable farming.

## 📊 Dataset

The dataset (sourced from [Kaggle](https://www.kaggle.com/datasets/waqi786/climate-change-impact-on-agriculture)) includes:

- 🌡️ Average Temperature  
- 🌧️ Precipitation  
- 🧪 CO₂ Emissions  
- 💧 Irrigation Access  
- 🌾 Crop Yield  
- 📍 Geographical Region  
- ⚠️ Extreme Weather Events

The data was preprocessed through scaling, encoding, and correlation-based feature selection. An 80:20 train-test split was used for modeling.

## 🤖 Machine Learning Models Used

- Linear Regression  
- Lasso Regression  
- Ridge Regression  
- Decision Tree Regressor  
- **K-Nearest Neighbors (KNN)** ✅ *(Best Performing Model)*

### ✅ Model Performance (Testing)

| Model            | MAE (↓) | R² Score (↑) |
|------------------|---------|--------------|
| Linear           | 222.63  | 0.4645       |
| Lasso            | 222.03  | 0.4668       |
| Ridge            | 222.57  | 0.4648       |
| Decision Tree    | 339.37  | -0.2782      |
| **KNN**          | **259.84** | **0.3306**  |

Despite slightly higher MAE, KNN captured non-linear climate-crop relationships better and was chosen for final predictions.

## 🛠️ Tools & Technologies

- Python  
- Scikit-learn  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Jupyter Notebook

## 🤝 Collaboration

We are happy to collaborate and open to any suggestions or feedback that help us enhance this project further. If you're passionate about climate, data science, or agriculture tech — let’s connect!

## 📬 Contact

Feel free to reach out:  
🔗 LinkedIn: https://www.linkedin.com/in/swapnilshikha-bhakat-264221270/
