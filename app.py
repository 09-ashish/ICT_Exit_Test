from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load files
model = pickle.load(open("pokemon_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')


@app.route('/result', methods=['POST'])
def result():

    height = float(request.form['height'])
    weight = float(request.form['weight'])
    exp = float(request.form['exp'])
    hp = float(request.form['hp'])
    attack = float(request.form['attack'])
    defense = float(request.form['defense'])

    data = np.array([
        [height, weight, exp, hp, attack, defense]
    ])

    data = scaler.transform(data)

    pred = model.predict(data)

    pokemon_type = encoder.inverse_transform(pred)[0]

    return render_template(
        'result.html',
        prediction=pokemon_type
    )


if __name__ == '__main__':
    app.run(debug=True)