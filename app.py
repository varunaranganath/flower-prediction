from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Load the Iris dataset and train the model
iris = load_iris()
X = iris.data
y = iris.target
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]).reshape(1, -1)
    prediction = knn.predict(features)
    response = {'prediction': iris.target_names[prediction[0]]}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)