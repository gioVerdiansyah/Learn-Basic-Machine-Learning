from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder="D:/Document/Learn Machine Learning/learn_4_lks/previews")

model = pickle.load(open("D:/Document/Learn Machine Learning/learn_4_lks/models/forecast_model.pkl", 'rb'))

@app.route('/')
def home():
  return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == "POST":
    input = [x for x in request.form.values()]
    final_output = [np.array(input)]
    prediction = model.predict(final_output)[0]
    return render_template('index.html', output=format(prediction.round(2)))

if __name__ == '__main__':
  app.run(debug=True)