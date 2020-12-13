from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


@dataclass
class Peak:
    center: float
    height: float
    width: float
    shape: str

    def __str__(self):
        return f"Peak in {self.center}, height {self.height} & width {self.width} of {self.shape} shape"

    def evaluate(self, ws):
        if self.shape == 'quadratic':
            values = self.height - (ws / self.width - self.center) ** 2
            # We forbid < 0 values
            return np.where(values < 0, 0, values)
        if self.shape == 'gaussian':
            width = self.width / 3  # to have similar width than quadratic function (ie function ~ 0 after width)
            return self.height * np.exp(- ((ws - self.center) / width)**2 / 2)
            # No  '/ (np.sqrt(2 * np.pi) * width)' as it will be renormalized later
        else:
            raise ValueError(f'Unexpected shape {self.shape} for Peak {self}')

    def plot(self, ws):
        values = self.evaluate(ws)
        plt.plot(ws, values, 'g')
        plt.plot(-ws, values, 'g')

    @classmethod
    def evaluate_all(cls, peaks, ws):
        return np.max([peak.evaluate(ws) for peak in peaks], axis=0)

    @staticmethod
    def integral(f, ws):
        dw = ws[1] - ws[0]  # assumed constant
        return np.sum(dw * f)

    @staticmethod
    def aggregate(peaks, ws):
        values = Peak.evaluate_all(peaks, ws)
        # Normalize (ToDo: investigate how to avoid this making go our main peak go below 1/2)
        return values / Peak.integral(values, ws)


class Evaluator:
    @staticmethod
    def show(model, X, y, ws, test_train):
        y_pred = model.predict(np.array([X, ]))[0]
        mae = metrics.mean_absolute_error(y, y_pred)
        loss = model.loss(y, y_pred)
        plt.title(
            f"Model prediction on a {test_train} sample\n"
            f"Loss = {loss:.5f} and MAE = {mae:.5f}"
        )
        plt.plot(ws, y, label='Target density function', color='yellowgreen')
        plt.plot(ws, y_pred, label='Predicted density function', color='orange')
        plt.ylabel('A($\omega$)')
        plt.xlabel('$\omega$')
        plt.show()

    @staticmethod
    def eval(model, X, y, indices, ws, test_train='train'):
        for i in indices:
            Evaluator.show(model, X[i], y[i], ws, test_train)
