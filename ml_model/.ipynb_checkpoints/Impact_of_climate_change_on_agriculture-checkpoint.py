import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats.mstats import winsorize

plt.style.use("ggplot")   #used to create plots-Grammar of Graphics
df=pd.read_csv("final_data.csv")
#print(df)
print(df.duplicated().sum())
# print(df.shape)
df = df[df['Country'] == 'India']
df=df.drop(columns='Country')  #to drop the column country

#scaling 
numeric_columns = df.select_dtypes(include=[np.number]).columns
numeric_columns = numeric_columns.difference(['Economic_Impact_Million_USD','Year'])
df[numeric_columns] = StandardScaler().fit_transform(df[numeric_columns])

plt.figure(figsize=(15,10))
sns.countplot(y = df['Crop_Type'])
# plt.show()

#storing the crop ,region names with the encoded values
crops_encoding=dict(enumerate(df['Crop_Type'].astype('category').cat.categories))
region_encoding=dict(enumerate(df['Region'].astype('category').cat.categories))
adaptation_encoding=dict(enumerate(df['Adaptation_Strategies'].astype('category').cat.categories))


#crops and region encoding
df['Crop_Type'] = df['Crop_Type'].astype('category').cat.codes
df['Region'] = df['Region'].astype('category').cat.codes
df['Adaptation_Strategies'] = df['Adaptation_Strategies'].astype('category').cat.codes




#if any missing values then replacing by mean
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)

correlation_matrix = df.corr(method='spearman')
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

df=df.drop(columns=['CO2_Emissions_MT','Pesticide_Use_KG_per_HA','Soil_Health_Index','Adaptation_Strategies'])


plt.figure(figsize=(15,10))
sns.histplot(df['Economic_Impact_Million_USD'], kde=True, bins=30)  # Histogram with a kernel density estimate (KDE)
plt.title("Distribution of Economic Impact (Million USD)")
plt.xlabel("Economic Impact (Million USD)")
plt.ylabel("Frequency")
# plt.show()

bins = [-np.inf, 10, 20, 30, 40, np.inf]
labels = ['<10°C', '10-20°C', '20-30°C', '30-40°C', '>40°C']
df['Temp_Range'] = df['Average_Temperature_C'].copy()
df['Temp_Range'] = pd.cut(df['Temp_Range'], bins=bins, labels=labels)
plt.figure(figsize=(15,10))
sns.barplot(x='Temp_Range', y='Crop_Yield_MT_per_HA', data=df, errorbar=None)
plt.title("Average Crop Yield per Temperature Range")
plt.xlabel("Temperature Range (°C)")
plt.ylabel("Average Crop Yield (MT per HA)")
# plt.show()

print(df.head())
print(region_encoding)
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

print(df)

y = df['Economic_Impact_Million_USD']
x = df.drop(['Economic_Impact_Million_USD','Temp_Range'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, shuffle=True)
print(x_train.dtypes)
sc=StandardScaler()
x_train_normalizaed=sc.fit_transform(x_train)
x_test_normalized=sc.transform(x_test)

print(x)
print(y)
models = {
    'Linear Regression' : LinearRegression(),
    'Lasso' : Lasso(),
    'Ridge' : Ridge(),
    'Decision Tree' : DecisionTreeRegressor(),
    'KNN' : KNeighborsRegressor()
}
for name, model in models.items():
    model.fit(x_train_normalizaed, y_train)
    y_train_pred = model.predict(x_train_normalizaed)

    # Calculate training error
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Print the training error
    print(f"{name} (Training): MAE: {train_mae}, R²: {train_r2}")

    # Predict on test data
    y_test_pred = model.predict(x_test_normalized)

    # Calculate test error (already implemented)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print the test error
    print(f"{name} (Testing): MAE: {test_mae}, R²: {test_r2}")

def prediction(
    Year,
    Average_Temperature_C,
    Region,
    Crop_Type,
    Total_Precipitation_mm,
    Crop_Yield_MT_per_HA,
    Extreme_Weather_Events,
    Irrigation_Access,
    Fertilizer_Use_KG_per_HA
):
    try:
        Region_encoded = list(region_encoding.keys())[list(region_encoding.values()).index(Region)]
        Crop_Type_encoded = list(crops_encoding.keys())[list(crops_encoding.values()).index(Crop_Type)]
    except ValueError as e:
        raise ValueError(f"Error encoding categorical inputs: {e}")
    features = pd.DataFrame([[
        Year,
        Average_Temperature_C,
        Region_encoded,
        Crop_Type_encoded,
        Total_Precipitation_mm,
        Crop_Yield_MT_per_HA,
        Extreme_Weather_Events,
        Irrigation_Access,
        Fertilizer_Use_KG_per_HA
    ]],columns=x.columns)
    features_scaled = sc.transform(features)

    rid=KNeighborsRegressor()
    rid.fit(x_train_normalizaed,y_train)

    # Predict the yield
    try:
        predicted_yield = rid.predict(features_scaled)
    except Exception as e:
        raise ValueError(f"Error during prediction: {e}")
    # Return the first prediction as a scalar
    return float(predicted_yield[0])
print(prediction(2001,1.55,"West Bengal","Corn",447.06,1.7369999999999999,8,14.54,10.08))
#808