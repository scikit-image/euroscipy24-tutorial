# Solution for exercise 3a
horiz_kernel = vertical_kernel.T

gradient_horiz = ndi.correlate(pixelated.astype(float),
                               horiz_kernel)

fig, ax = plt.subplots(ncols=2)
ax[0].imshow(gradient_vertical, cmap="gray");
ax[1].imshow(gradient_horiz, cmap="gray");