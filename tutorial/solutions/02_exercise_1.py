# Solution for exercise 1
import scipy as sp

smoothed_with_scipy = sp.ndimage.convolve(noisy_signal, mean_kernel11, mode="reflect")

fig, ax = plt.subplots()
ax.plot(smooth_signal11same, label="padding with zeros")
ax.plot(smoothed_with_scipy, label="mode reflect")
ax.legend()