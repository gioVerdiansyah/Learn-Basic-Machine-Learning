from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder="D:/Document/Learn Machine Learning/learn_4_lks/exam/preview")

model = pickle.load(
    open("/learn_4_lks/exam/models/classification_test.pkl", "rb")
)

@app.route("/")
def index():
    return render_template("classification_test.html")


@app.route('/predict', methods=['POST'])
def predict():
    items = dict(request.form.items())
    short_cols = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
    shorted_items = {}
    for col in short_cols:
        shorted_items[col] = items[col]
    print(shorted_items)
    print(list(shorted_items.values()))

    predict_val = model.predict([np.array(list(shorted_items.values()))])[0]
    print("PREDICTION")
    print(predict_val)

    result = "Anda berpotensi TERKENA Stroke!" if predict_val == 1 else "Alhamdullilah Anda berpotensi TIDAK terkena Stroke!"
    return render_template("classification_test_result_page.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)