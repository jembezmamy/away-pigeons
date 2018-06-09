from vendor.MeanShift_py import mean_shift as ms
import numpy as np

def detect(predictions, bandwidth = 1):
    points = np.array([np.array([x * 1.0, y * 1.0]) for prediction, x, y in predictions if prediction > 0.5])
    mean_shifter = ms.MeanShift()
    return mean_shifter.cluster(points, kernel_bandwidth = bandwidth)
