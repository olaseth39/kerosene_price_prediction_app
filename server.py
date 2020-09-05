from flask import Flask, request, jsonify, render_template, json
import numpy as np, pickle
#from wtforms import Form, StringField, IntegerField, validators

app = Flask(__name__)

#get the month names
with open("model/columns.json", "r") as f:
        data_columns = json.load(f)['data_columns']
        months = data_columns[1:]      # start from the month name

#get the pickled model
with open('model/kerosene_price_prediction.pickle', 'rb') as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template('app.html')


@app.route('/predict_kero_price', methods=['POST'])
def predict_kero_price():
    values = [i for i in request.form.values()]
    year = values[0]     #get the year
    #year = int(v1)     #change to integer
    #print(type(year))
    month = values[1]     #get the month
    #print(year)

    try:
        month_index = data_columns.index(month.lower())

    except:
        month_index = -1 # where month_index is not found

    x = np.zeros(len(data_columns))
    x[0] = year

    if month_index >= 0:
        x[month_index] = 1
    
    predicted_price = round(model.predict([x])[0], 2)
    #print(round(model.predict([x])[0], 2))
    

    return render_template('app.html', predicted_price=predicted_price)

   
if __name__ == "__main__":
    print("Starting Python Flask Server For Kerosene Price Prediction")
    app.run(debug=True)

