from flask import Flask, render_template, request
import jsonify
import requests
import pandas as pd
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = pickle.load(open('automodel_lasso_6cols.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

# columns = ['hp_kw', 'km', 'age', 'make_model_Audi A1', 'make_model_Audi A3',
#            'make_model_Opel Astra', 'make_model_Opel Corsa',
#            'make_model_Opel Insignia', 'make_model_Renault Clio',
#            'make_model_Renault Duster', 'make_model_Renault Espace',
#            'gearing_type_Automatic', 'gearing_type_Manual',
#            'gearing_type_Semi-automatic']  # our dummied columns


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age']) # Age
        km = int(request.form['km']) # Km
        hp_kw = int(request.form['hp_kw']) # hp_kW
        
        # MAKE_MODEL
        make_model = request.form["make_model"]
        if(make_model == 'Audi_A1'):
            make_model_Audi_A1 = 1
            make_model_Audi_A3 = 0
            make_model_Opel_Astra = 0
            make_model_Opel_Corsa = 0
            make_model_Opel_Insignia = 0
            make_model_Renault_Clio = 0
            make_model_Renault_Duster = 0
            make_model_Renault_Espace = 0
                
        elif(make_model == 'Audi_A3'):
            make_model_Audi_A1 = 0
            make_model_Audi_A3 = 1
            make_model_Opel_Astra = 0
            make_model_Opel_Corsa = 0
            make_model_Opel_Insignia = 0
            make_model_Renault_Clio = 0
            make_model_Renault_Duster = 0
            make_model_Renault_Espace = 0

        elif(make_model == 'Opel_Astra'):
            make_model_Audi_A1 = 0
            make_model_Audi_A3 = 0
            make_model_Opel_Astra = 1
            make_model_Opel_Corsa = 0
            make_model_Opel_Insignia = 0
            make_model_Renault_Clio = 0
            make_model_Renault_Duster = 0
            make_model_Renault_Espace = 0

        elif(make_model == 'Opel_Corsa'):
            make_model_Audi_A1 = 0
            make_model_Audi_A3 = 0
            make_model_Opel_Astra = 0
            make_model_Opel_Corsa = 1
            make_model_Opel_Insignia = 0
            make_model_Renault_Clio = 0
            make_model_Renault_Duster = 0
            make_model_Renault_Espace = 0

        elif(make_model == 'Opel_Insignia'):
            make_model_Audi_A1 = 0
            make_model_Audi_A3 = 0
            make_model_Opel_Astra = 0
            make_model_Opel_Corsa = 0
            make_model_Opel_Insignia = 1
            make_model_Renault_Clio = 0
            make_model_Renault_Duster = 0
            make_model_Renault_Espace = 0

        elif(make_model == 'Renault_Clio'):
            make_model_Audi_A1 = 0
            make_model_Audi_A3 = 0
            make_model_Opel_Astra = 0
            make_model_Opel_Corsa = 0
            make_model_Opel_Insignia = 0
            make_model_Renault_Clio = 1
            make_model_Renault_Duster = 0
            make_model_Renault_Espace = 0


        elif(make_model=='Renault_Duster'):
            make_model_Audi_A1 = 0
            make_model_Audi_A3 = 0
            make_model_Opel_Astra = 0
            make_model_Opel_Corsa = 0
            make_model_Opel_Insignia = 0
            make_model_Renault_Clio = 0
            make_model_Renault_Duster = 1
            make_model_Renault_Espace = 0

        else:
            make_model_Audi_A1 = 0
            make_model_Audi_A3 = 0
            make_model_Opel_Astra = 0
            make_model_Opel_Corsa = 0
            make_model_Opel_Insignia = 0
            make_model_Renault_Clio = 0
            make_model_Renault_Duster = 0
            make_model_Renault_Espace = 1

        # GEARING_TYPE
        gearing_type = request.form["gearing_type"]
        if(gearing_type == 'Automatic'):
            gearing_type_Automatic = 1
            gearing_type_Manual = 0
            gearing_type_Semi_Automatic = 0

        elif(gearing_type == "Manual"):
            gearing_type_Automatic = 0
            gearing_type_Manual = 1
            gearing_type_Semi_Automatic = 0

        else:
            gearing_type_Automatic = 0
            gearing_type_Manual = 0
            gearing_type_Semi_Automatic = 1

        
        df_new = pd.read_csv("final_scout_dummy.csv")
        X = df_new.drop(columns = ["price"])
        final_scaler = MinMaxScaler()
        final_scaler.fit(X)
        
        
        mylist = [[hp_kw, km, age, make_model_Audi_A1 ,make_model_Audi_A3, make_model_Opel_Astra, make_model_Opel_Corsa, make_model_Opel_Insignia, make_model_Renault_Clio, make_model_Renault_Duster, make_model_Renault_Espace, gearing_type_Automatic, gearing_type_Manual, gearing_type_Semi_Automatic]]
        data = pd.DataFrame(mylist, columns= ["hp_kw", "km", "age", "make_model_Audi_A1" ,"make_model_Audi_A3", "make_model_Opel_Astra", "make_model_Opel_Corsa", "make_model_Opel_Insignia", "make_model_Renault_Clio", "make_model_Renault_Duster", "make_model_Renault_Espace", "gearing_type_Automatic", "gearing_type_Manual", "gearing_type_Semi_Automatic"])

        data = final_scaler.transform(data)
        prediction = model.predict(data)
        
        return render_template("index.html",prediction_text="You Can Sell The Car at {}".format(round(prediction[0],2)))

    else:
        return render_template('index.html')



if __name__=="__main__":
    app.run(debug=True)
