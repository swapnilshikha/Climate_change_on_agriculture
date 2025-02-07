# from flask import Flask,request, render_template
# import numpy as np
# import pickle
# import sklearn
# print(sklearn.__version__)
# #loading models
# dtr = pickle.load(open('models/knn.pkl','rb'))
# preprocessor = pickle.load(open('models/preprocesser.pkl','rb'))

# #flask app
# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')
# @app.route("/predict",methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             Temp = float(request.form['Temp'])
#             Region = request.form['Region']
#             crop = request.form['crop']
#             precip = float(request.form['precip'])
#             crop_yield = float(request.form['crop_yield'])
#             weather = int(request.form['weather'])
#             irrigation = float(request.form['irrigation'])
#             fertilizer = float(request.form['fertilizer'])

#             features = np.array([[Temp,Region,crop,precip, crop_yield,weather,irrigation,fertilizer]],dtype=object)
#             transformed_features = preprocessor.transform(features)
#             # prediction = dtr.predict(transformed_features).reshape(1,-1)
#             prediction = dtr.predict(transformed_features)[0]
#             # return render_template('index.html',prediction = prediction[0][0])
#             return render_template('index.html', prediction=prediction)
#         except ValueError:
#             return render_template('index.html', error="Invalid input. Please enter numeric values for Year, Rainfall, and Pesticides.")
#         except Exception as e:
#             return render_template('index.html', error=f"Error during prediction: {e}")

# if __name__=="__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)

# Load models
dtr = pickle.load(open('models/knn.pkl', 'rb'))
preprocessor = pickle.load(open('models/preprocesser.pkl', 'rb'))

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            Avg_Temp = float(request.form['Temp'])
            Region = request.form['Region']
            crop = request.form['crop']
            precip = float(request.form['precip'])
            crop_yield = float(request.form['crop_yield'])
            weather = int(request.form['weather'])
            irrigation = float(request.form['irrigation'])
            fertilizer = float(request.form['fertilizer'])

            # Preprocess inputs
            features = np.array([[Avg_Temp, Region, crop, precip, crop_yield, weather, irrigation, fertilizer]], dtype=object)
            transformed_features = preprocessor.transform(features)
            
            # Make prediction
            prediction = dtr.predict(transformed_features)[0]
            return render_template('index.html', prediction=prediction)
        except ValueError:
            return render_template('index.html', error="Invalid input. Please enter valid values.")
        except Exception as e:
            return render_template('index.html', error=f"Error during prediction: {e}")

if __name__ == "__main__":
    app.run(debug=True)
