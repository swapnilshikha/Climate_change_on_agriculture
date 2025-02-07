from flask import Flask,request, render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)
#loading models
dtr = pickle.load(open('models/knn.pkl','rb'))
preprocessor = pickle.load(open('models/preprocesser.pkl','rb'))

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            Year = int(request.form['Year'])
            annual_rainfall = float(request.form['Annual_Rainfall'])
            pesticides = float(request.form['Pesticides'])
            state = request.form['State']
            crop = request.form['Crop']

            features = np.array([[Year,annual_rainfall,pesticides,state, crop]],dtype=object)
            transformed_features = preprocessor.transform(features)
            # prediction = dtr.predict(transformed_features).reshape(1,-1)
            prediction = dtr.predict(transformed_features)[0]

            # return render_template('index.html',prediction = prediction[0][0])
            return render_template('index.html', prediction=prediction)
        except ValueError:
            return render_template('index.html', error="Invalid input. Please enter numeric values for Year, Rainfall, and Pesticides.")
        except Exception as e:
            return render_template('index.html', error=f"Error during prediction: {e}")

if __name__=="__main__":
    app.run(debug=True)