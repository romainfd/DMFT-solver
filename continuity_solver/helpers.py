from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Peak:
    center: float
    height: float
    width: float
    shape: str

    def __str__(self):
        return f"Peak in {self.center}, height {self.height} & width {self.width} of {self.shape} shape"

    def evaluate(self, ws):
        values = self.height - (ws / self.width - self.center) ** 2
        # We forbid < 0 values
        return np.where(values < 0, 0, values)

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
        # Normalize (ToDo: investigate as it can make us go below 1/2)
        return values / Peak.integral(values, ws) / 2 # since we only focus on w >= 0)


class Evaluator:
    @staticmethod
    def show(model, X, y, ws):
        y_pred = model.predict(np.array([X, ]))[0]
        plt.plot(ws, y)
        plt.plot(ws, y_pred)
        plt.show()
