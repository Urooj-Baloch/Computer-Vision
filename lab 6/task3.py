import numpy as np
import pywt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
n = 1000
t = np.arange(n)
sensor = np.random.normal(0, 1, n)
sensor += 0.5 * np.sin(2 * np.pi * t / 100)
spike_idx = np.random.choice(n, 30, replace=False)
sensor[spike_idx] += np.random.choice([-1, 1], 30) * np.random.uniform(1.5, 2.5, 30)

wavelet = "db4"
level = 4
coeffs = pywt.wavedec(sensor, wavelet, level=level)
sigma = np.median(np.abs(coeffs[-1])) / 0.6745
threshold = sigma * np.sqrt(2 * np.log(len(sensor))) * 2
denoised_coeffs = [coeffs[0] * 0] + [pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]]
denoised = pywt.waverec(denoised_coeffs, wavelet)[:n]

residuals = sensor - denoised
res_thresh = np.mean(residuals) + 2 * np.std(residuals)
anomalies = np.where(np.abs(residuals - np.mean(residuals)) > 2 * np.std(residuals))[0]

print("Total points:", n)
print("Anomalies detected:", len(anomalies))
print("Anomaly indices:", anomalies.tolist())

fig, axes = plt.subplots(2, 1, figsize=(10, 7))
axes[0].plot(sensor, label="Sensor Data", color="tab:blue")
axes[0].plot(denoised, label="Denoised Signal", color="tab:orange", linestyle="--")
axes[0].set_title("Sensor Data and Denoised Signal")
axes[0].legend(loc="lower right")

axes[1].plot(residuals, label="Residuals", color="red")
axes[1].scatter(anomalies, residuals[anomalies], color="green", label="Anomalies", zorder=3)
axes[1].set_title("Residuals and Detected Anomalies")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.savefig("output/task3_anomalies.png", dpi=150)
