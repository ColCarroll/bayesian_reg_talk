import os
import numpy
import numpy.random
from sklearn import linear_model
from flask import Flask, jsonify, request, redirect, url_for

DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_url_path='')
FUNCS = [
    lambda x: 1 - 0.2 * x,
    lambda x: x * x,
    lambda x: numpy.sin(5 * x),
    lambda x: numpy.exp(x),
    lambda x: 1. / (1 + numpy.exp(-5 * x)),
    lambda x: numpy.sin(1. / x),
]


def normalize(mat):
    return mat / abs(mat).sum(0)


class Model:
    def __init__(self, **kwargs):
        self._func = FUNCS[kwargs.get("func_no", 1)]
        self._noise = kwargs.get("noise", 0.05)
        self._model_type = kwargs.get("model_type", "least_squares")
        self._normalize = kwargs.get("normalize", False)
        self._alpha = kwargs.get("alpha", 0.01)
        self._basis_type = kwargs.get("basis_type", "polynomial")
        self._num_basis = kwargs.get("num_basis", 3)
        self._training_points = kwargs.get("training_points", 30)
        self._testing_points = kwargs.get("testing_points", 100)
        self._model = linear_model.LinearRegression()
        self._features = {}
        self._data = {}

        self.responses = {
            "func_no": lambda j: self.update_func(int(j)),
            "noise": lambda j: self.update_noise(float(j)),
            "model_type": self.update_model_type,
            "normalize": lambda j: self.update_normalize(j == "true"),
            "alpha": lambda j: self.update_alpha(float(j)),
            "basis_type": self.update_basis_type,
            "num_basis": lambda j: self.update_num_basis(int(j)),
            "training_points": lambda j: self.update_training_points(int(j)),
            "testing_points": lambda j: self.update_testing_points(int(j))
        }
        self.update_data()

    def update(self, request_dict):
        for key, key_func in self.responses.iteritems():
            if key in request_dict:
                key_func(request_dict[key])
                break

    def update_func(self, new_func):
        self._func = FUNCS[new_func]
        self.update_data()

    def update_noise(self, new_noise):
        self._noise = new_noise
        self.update_data()

    def update_model_type(self, new_model_type):
        self._model_type = new_model_type
        self.update_model()

    def update_normalize(self, new_normalize):
        self._normalize = new_normalize
        self.update_features()

    def update_alpha(self, new_alpha):
        self._alpha = new_alpha
        self.update_model()

    def update_basis_type(self, new_basis_type):
        self._basis_type = new_basis_type
        self.update_features()

    def update_num_basis(self, new_num_basis):
        self._num_basis = new_num_basis
        self.update_features()

    def update_training_points(self, new_training_points):
        self._training_points = new_training_points
        self.update_data()

    def update_testing_points(self, new_testing_points):
        self._testing_points = new_testing_points
        self.update_data()

    def update_data(self):
        self._data = {
            "x": {
                "plot": numpy.linspace(-1, 1, 250),
                "train": numpy.sort(2 * numpy.random.random(self._training_points) - 1),
                "test": numpy.sort(2 * numpy.random.random(self._testing_points) - 1),
            }}
        self._data["y"] = {key: self._func(value) + self._noise * numpy.random.normal(0, 1, len(value)) for key, value
                           in self._data["x"].iteritems()}
        self.update_features()

    def update_features(self):
        self._features = {}
        if self._basis_type == "polynomial":
            for key, value in self._data["x"].iteritems():
                self._features[key] = numpy.array([numpy.power(value, j) for j in range(1, self._num_basis + 1)]).T
                if self._normalize:
                    self._features[key] = normalize(self._features[key])
        self.fit_model()

    def update_model(self):
        if self._model_type == "ridge":
            self._model = linear_model.Ridge(self._alpha)
        elif self._model_type == "lasso":
            self._model = linear_model.Lasso(self._alpha)
        else:
            self._model = linear_model.LinearRegression()
        self.fit_model()

    def fit_model(self):
        self._data["y_pred"] = {}
        self._model.fit(self._features["train"], self._data["y"]["train"])
        for data_set in self._features.iterkeys():
            self._data["y_pred"][data_set] = self._model.predict(self._features[data_set])
        self.update_errors()

    def update_errors(self):
        self._data["errors"] = {}
        for data_set in self._features.iterkeys():
            self._data["errors"][data_set] = numpy.power(
                self._data["y_pred"][data_set] - self._data["y"][data_set], 2
            ).mean()

    def json(self):
        json_data = {key: zip(self._data["x"][key], self._data["y"][key]) for key in ("train", "test")}
        json_data["plot"] = zip(self._data["x"]["plot"], self._data["y_pred"]["plot"])
        json_data["errors"] = self._data["errors"]
        return json_data


MODEL = Model()


@app.route('/')
def index():
    return app.send_static_file("index.html")


@app.route('/update')
def update():
    MODEL.update(request.args)
    return redirect(url_for("get"))

@app.route('/data')
def get():
    return jsonify(MODEL.json())


if __name__ == '__main__':
    app.run(debug=True, port=8080)