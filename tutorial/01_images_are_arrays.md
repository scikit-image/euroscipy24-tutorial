---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python (Pyodide)
    language: python
    name: python
---

```python tags=["remove-input", "remove-output"]
%config InlineBackend.figure_format = 'retina'
```

```python
import numpy as np
print(f'NumPy {np.__version__}')

import skimage as ski
print(f'skimage {ski.__version__}')
```

*Note: NumPy 2.0 was released recently; scikit-image 0.24 is compatible.*


⚠️ Note the import convention above: `import skimage as ski`.
If you don't import `skimage`, the examples below will fail to execute.


# Part 1: Images are numpy arrays


Images are represented in ``scikit-image`` using standard ``numpy`` arrays.  This allows maximum interoperability with other libraries in the scientific Python ecosystem, such as ``matplotlib`` and ``scipy``.

Let's build a grayscale image as a 2D array:

```python
rng = np.random.default_rng(44)
random_image = rng.random([500, 500])
random_image.shape, random_image.dtype
```

```python
import matplotlib.pyplot as plt

plt.imshow(random_image, cmap='gray')
plt.colorbar();
```

The value of `random_image[i, j]` is the intensity value of the pixel located at coordinates `[i, j]`.
The same holds for "real-world" images:

```python
coins = ski.data.coins()

print('Type:', type(coins))
print('dtype:', coins.dtype)
print('shape:', coins.shape)

plt.imshow(coins, cmap='gray');
```

A color image is a 3D array, where the last dimension has length 3 and contains the red, green, and blue channels:

```python
cat = ski.data.chelsea()

print("Shape:", cat.shape)
print("Values min/max:", cat.min(), cat.max())

plt.imshow(cat);
```

These are *just NumPy arrays*. E.g., we can make a red square by using standard array slicing and manipulation:

```python
cat[10:110, 10:110, :] = [255, 0, 0]  # [red, green, blue]
plt.imshow(cat);
```

Images can also include transparency, which is represented by a 4th channel, called the *alpha layer*.


## Other shapes, and their meanings

|Image type|Coordinates|
|:---|:---|
|2D grayscale|(row, column)|
|2D multichannel|(row, column, channel)|
|3D grayscale (or volumetric) |(plane, row, column)|
|3D multichannel|(plane, row, column, channel)|

See [Coordinate conventions](https://scikit-image.org/docs/stable/user_guide/numpy_images.html#coordinate-conventions).


## Data types and image values

There are different conventions for representing image intensity values, the most common ones being:

```
  0 - 255   where  0 is black, 255 is white
  0 - 1     where  0 is black, 1 is white
```

``scikit-image`` supports both conventions—the choice is determined by the
data type (`dtype`) of the array.

E.g., here, we generate two equally valid images:

```python
linear0 = np.linspace(0, 1, 2500).reshape((50, 50))
linear1 = np.linspace(0, 255, 2500).reshape((50, 50)).astype(np.uint8)

print("Linear0:", linear0.dtype, linear0.min(), linear0.max())
print("Linear1:", linear1.dtype, linear1.min(), linear1.max())

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 15))
ax0.imshow(linear0, cmap='gray')
ax1.imshow(linear1, cmap='gray');
```

When we first designed the library, we *assumed* that floating-point type images would always range from 0 to 1, and unsigned integers from 0 to 255.

We are moving away from that design, because often you will find quantities that don't fit that mold (temperature or rainfall data, fluorescence microscopy images, ...).

If you're working with standard imaging data, continue to use 0-1.

When loading integer images (e.g., 0-255), you'll often want to convert those to floating point (0-1). You may do that using `img_as_float`:

```python
cat = ski.data.chelsea()
print(cat.dtype, cat.min(), cat.max())

cat_float = ski.util.img_as_float(cat)
print(cat_float.dtype, cat_float.min(), cat_float.max())

print()
print("cat.max()/255 =", cat.max()/255)
```

More at https://scikit-image.org/docs/stable/user_guide/data_types.html.


## Image I/O

Usually, we don't use images from the scikit-image example datasets, but images stored on disk in JPEG, PNG, or TIFF format. Since scikit-image operates on NumPy arrays, *any* image reader library that returns arrays will do. We recommend `imageio`, but `matplotlib`, `pillow`, etc. also work.

scikit-image provides a convenience wrapper around `imageio`, in the form of the `skimage.io` submodule:

```python
image = ski.io.imread('data/balloon.jpg')

print(type(image))
print(image.dtype)
print(image.shape)
print(image.min(), image.max())

plt.imshow(image);
```


## Exercise 1: draw the letter H

Define a function that takes as input an RGB image (shape `MxNx3`) and a pair of coordinates, `(row, column)`, and returns a copy with a green letter H overlaid at those coordinates. The coordinates point to the top-left corner of the H.

The arms and strut of the H should have a thickness of 3 pixels, and the H itself should have a height of 24 pixels and width of 20 pixels.

Start with the following template:

```python tags=["hide-output"]
def draw_H(image, coords, color=(0, 255, 0)):
    out = image.copy()
    
    ...
    
    return out 
```

Test your function like so:

```python tags=["remove-output"]
cat = ski.data.chelsea()
cat_H = draw_H(cat, (50, -50))
plt.imshow(cat_H);
```

## <span class="exercize">Exercise 2: visualizing RGB channels</span>

Display the different color channels of the image (each as a gray-scale image).  Start with the following template:

```python tags=["raises-exception", "remove-output"]
# --- read in the image ---

image = plt.imread('./data/Bells-Beach.jpg')

# --- assign each color channel to a different variable ---

r = ...  # FIXME: grab channel from image...
g = ...  # FIXME
b = ...  # FIXME

# --- display the image and r, g, b channels ---
f, axes = plt.subplots(1, 4, figsize=(16, 5))

for ax in axes:
    ax.axis('off')

(ax_r, ax_g, ax_b, ax_color) = axes

ax_r.imshow(r, cmap='gray')
ax_r.set_title('red channel')

ax_g.imshow(g, cmap='gray')
ax_g.set_title('green channel')

ax_b.imshow(b, cmap='gray')
ax_b.set_title('blue channel')

# --- Here, we stack the R, G, and B layers again
#     to form a color image ---
ax_color.imshow(np.stack([r, g, b], axis=2))
ax_color.set_title('all channels');
```

Now, take a look at the following R, G, and B channels.  How would their combination look? (Write some code to confirm your intuition.)

```python
red = np.zeros((300, 300))
green = np.zeros((300, 300))
blue = np.zeros((300, 300))

r, c = ski.draw.disk(center=(100, 100), radius=100)
red[r, c] = 1

r, c = ski.draw.disk(center=(100, 200), radius=100)
green[r, c] = 1

r, c = ski.draw.disk(center=(200, 150), radius=100)
blue[r, c] = 1

f, axes = plt.subplots(1, 3)
for (ax, channel, name) in zip(axes, [red, green, blue], ['red', 'green', 'blue']):
    ax.imshow(channel, cmap='gray')
    ax.set_title(f'{name} channel')
    ax.axis('off')
```

## Exercise 3: Convert to grayscale ("black and white")

The *relative luminance* of an image is the intensity of light coming from each point. Different colors contribute differently to the luminance: it's very hard to have a bright, pure blue, for example. So, starting from an RGB image, the luminance is given by:

$$
Y = 0.2126R + 0.7152G + 0.0722B
$$

Use Python 3.5's matrix multiplication, `@`, to convert an RGB image to a grayscale (luminance) image according to the formula above.

Compare your results to that obtained with `skimage.color.rgb2gray`.

Change the coefficients above to be 1/3 (i.e., take the mean of the red, green, and blue channels), to see how that approach compares with `rgb2gray`.

```python tags=["raises-exception", "remove-output"]
image = ski.img_as_float(ski.io.imread('./data/balloon.jpg'))

gray = ski.color.rgb2gray(image)
my_gray = ...  # FIXME; compute R, G, B average here

# --- display the results ---
f, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 6))

ax0.imshow(gray, cmap='gray')
ax0.set_title('skimage.color.rgb2gray')

ax1.imshow(my_gray, cmap='gray')
ax1.set_title('my rgb2gray')
```
