# Solution for exercise 3b
gradient = np.sqrt(gradient_vertical**2 + gradient_horiz**2)

fig, ax = plt.subplots()
im = ax.imshow(gradient, cmap="gray");
fig.colorbar(im)