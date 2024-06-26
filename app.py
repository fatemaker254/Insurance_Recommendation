from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os

app = Flask(__name__)

# Load the dataset
riz = pd.read_csv(r"insurance.csv")

riz.loc[:, "sex"].replace({"male": 0, "female": 1}, inplace=True)
riz.loc[:, "smoker"].replace({"yes": 0, "no": 1}, inplace=True)
riz.loc[:, "region"].replace(
    {"southeast": 0, "southwest": 1, "northeast": 2, "northwest": 3}, inplace=True
)


# Splitting the Features and Target
X = riz.drop(columns="charges", axis=1)
Y = riz["charges"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_data = []
    input_data.append(int(request.form["age"]))
    input_data.append(int(request.form["sex"]))
    input_data.append(float(request.form["bmi"]))
    input_data.append(int(request.form["children"]))
    input_data.append(int(request.form["smoker"]))
    input_data.append(int(request.form["region"]))

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = regressor.predict(input_data_reshaped)
    inr = prediction[0]
    return render_template("index.html", prediction=inr, show_result=True)
    # return f"The insurance cost is (INR {inr:.2f})"


@app.route("/data", methods=["GET", "POST"])
def data():
    if request.method == "POST":
        new_data = {
            "age": int(request.form["age"]),
            "sex": int(request.form["sex"]),
            "bmi": float(request.form["bmi"]),
            "children": int(request.form["children"]),
            "smoker": int(request.form["smoker"]),
            "region": int(request.form["region"]),
            "Disease": str(request.form["Disease"]),
        }
        new_df = pd.DataFrame(new_data, index=[0])
        new_df.to_csv(
            "new_data.csv",
            mode="a",
            index=False,
            header=not os.path.exists("new_data.csv"),
        )
        return render_template("data.html", message="Data added successfully!")
    return render_template("data.html", message=None)


if __name__ == "__main__":
    app.run(debug=True)
