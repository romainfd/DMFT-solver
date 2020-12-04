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