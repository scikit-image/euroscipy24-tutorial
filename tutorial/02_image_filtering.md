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

```python tags=["hide-cell"]
%pip install -q ipywidgets ipympl
%matplotlib widget
%config InlineBackend.figure_format = 'retina'
```

```python slideshow={"slide_type": "skip"}
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
```

<!-- #region slideshow={"slide_type": "slide"} -->
# Part 2: Image filtering
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
Filtering is one of the most basic and common image operations in image processing. You can filter an image to remove noise or to enhance features; the filtered image could be the desired result or just a preprocessing step. Regardless, filtering is an important topic to understand.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
## Local filtering in one dimension
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
The "local" in local filtering simply means that a pixel is adjusted by values in some surrounding neighborhood. These surrounding elements are identified or weighted based on a "footprint", "structuring element", or "kernel".

Let's go to back to basics and look at a 1D step-signal
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
step_signal = np.zeros(100)
step_signal[50:] = 1

fig, ax = plt.subplots()
ax.plot(step_signal)
ax.margins(y=0.1)
```

<!-- #region slideshow={"slide_type": "notes"} -->
Next we add some noise to this signal:
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
# Just to make sure we all see the same results
rng = np.random.default_rng(44)
noise = rng.normal(0, 0.35, step_signal.shape)

noisy_signal = step_signal + noise

fig, ax = plt.subplots()
ax.plot(noisy_signal);
```

<!-- #region editable=true slideshow={"slide_type": "notes"} -->
The simplest way to recover something that looks a bit more like the original signal is to take the average between neighboring "pixels":
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
# Take the mean of neighboring pixels
smooth_signal = (noisy_signal[:-1] + noisy_signal[1:]) / 2.0

fig, ax = plt.subplots()
ax.plot(smooth_signal);
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
What happens if we want to take the *three* neighboring pixels? We can do the same thing:
<!-- #endregion -->

```python
smooth_signal3 = (noisy_signal[:-2]
                  + noisy_signal[1:-1]
                  + noisy_signal[2:]) / 3

fig, ax = plt.subplots()
ax.plot(noisy_signal, alpha=0.5, label="original")
ax.plot(smooth_signal, linestyle="dashed", label='mean of 2')
ax.plot(smooth_signal3, label='mean of 3')
ax.legend(loc='upper left');
```

<!-- #region editable=true slideshow={"slide_type": "notes"} -->
For averages of more points, the expression keeps getting hairier. And you have to worry more about what's going on in the margins. Is there a better way?

It turns out there is. This same concept, nearest-neighbor averages, can be expressed as a *convolution* with an *averaging kernel*. Note that the operation we did with `smooth_signal3` can be expressed as follows:

* Create an output array called `smooth_signal3`, of the same length as `noisy_signal`.
* At each element in `smooth_signal3` starting at point 1, and ending at point -2, place the average of the sum of: 1/3 of the element to the left of it in `noisy_signal`, 1/3 of the element at the same position, and 1/3 of the element to the right.
* discard the leftmost and rightmost elements.

This is called a *convolution* between the input image and the array `[1/3, 1/3, 1/3]`. (We'll give a more in-depth explanation of convolution in the next section).
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
# Same as above, using a convolution kernel
# Neighboring pixels multiplied by 1/3 and summed
mean_kernel3 = np.full((3,), 1/3)
smooth_signal3p = np.convolve(noisy_signal, mean_kernel3,
                              mode='valid')
fig, ax = plt.subplots()
ax.plot(smooth_signal3p)

print('smooth_signal3 and smooth_signal3p are equal:',
      np.allclose(smooth_signal3, smooth_signal3p))
```

```python editable=true slideshow={"slide_type": ""}
def convolve_demo(signal, kernel):
    ksize = len(kernel)
    convolved = np.correlate(signal, kernel)
    fig, ax = plt.subplots()
    
    def filter_step(i):
        ax.clear()
        ax.plot(signal, label='signal')
        ax.plot(convolved[:i+1], label='convolved')
        ax.legend()
        ax.scatter(np.arange(i, i+ksize),
                   signal[i : i+ksize])
        ax.scatter(i, convolved[i])
        plt.show()
    return filter_step

from ipywidgets import interact, widgets

i_slider = widgets.IntSlider(min=0, max=len(noisy_signal) - 3,
                             value=0)

interact(convolve_demo(noisy_signal, mean_kernel3),
         i=i_slider);
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
The advantage of convolution is that it's just as easy to take the average of 11 points as 3:
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
mean_kernel11 = np.full((11,), 1/11)
smooth_signal11 = np.convolve(noisy_signal, mean_kernel11,
                              mode='valid')
fig, ax = plt.subplots()
ax.plot(smooth_signal11);
```

```python
i_slider = widgets.IntSlider(min=0, max=len(noisy_signal) - 11,
                             value=0)

interact(convolve_demo(noisy_signal, mean_kernel11),
         i=i_slider);
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
Of course, to take the mean of 11 values, we have to move further and further away from the edges, and this starts to be noticeable. You can use `mode='same'` to pad the edges of the array and compute a result of the same size as the input:
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
smooth_signal3same = np.convolve(noisy_signal, mean_kernel3,
                                 mode='same')
smooth_signal11same = np.convolve(noisy_signal, mean_kernel11,
                                  mode='same')

fig, ax = plt.subplots(1, 2)
ax[0].plot(smooth_signal3p)
ax[0].plot(smooth_signal11)
ax[0].set_title('mode=valid')
ax[1].plot(smooth_signal3same)
ax[1].plot(smooth_signal11same)
ax[1].set_title('mode=same');
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
But now we see edge effects on the ends of the signal...

This is because `mode='same'` actually pads the signal with 0s and then applies `mode='valid'` as before.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
def convolve_demo_same(signal, kernel):
    ksize = len(kernel)
    padded_signal = np.pad(signal, ksize // 2,
                           mode='constant')
    convolved = np.correlate(padded_signal, kernel)
    fig, ax = plt.subplots()

    def filter_step(i):
        ax.clear()
        x = np.arange(-ksize // 2,
                      len(signal) + ksize // 2)
        ax.plot(signal, label='signal')
        ax.plot(convolved[:i+1], label='convolved')
        ax.legend()
        start, stop = i, i + ksize
        ax.scatter(x[start:stop]+1,
                   padded_signal[start : stop])
        ax.scatter(i, convolved[i])
        ax.set_xlim(-ksize // 2,
                    len(signal) + ksize // 2)
        plt.show()
    return filter_step


i_slider = widgets.IntSlider(min=0, max=len(noisy_signal)-1,
                             value=0)

interact(convolve_demo_same(noisy_signal, mean_kernel11),
         i=i_slider);
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
### Exercise 1:

Look up the documentation of `scipy.ndimage.convolve`. Apply the same convolution, but using a different `mode=` keyword argument to avoid the edge effects we see here.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
# Solution here
```

<!-- #region slideshow={"slide_type": "slide"} -->
### A difference filter
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "notes"} -->
Let's look again at our simplest signal, the step signal from before:
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
fig, ax = plt.subplots()
ax.plot(step_signal)
ax.margins(y=0.1) 
```

### Exercise 2:

Can you predict what a convolution with the kernel `[-1, 0, 1]` does? Try thinking about it before running the cells below.

```python editable=true slideshow={"slide_type": "fragment"}
result_corr = np.correlate(step_signal, np.array([-1, 0, 1]),
                           mode='valid')
```

```python
result_conv = np.convolve(step_signal, np.array([-1, 0, 1]),
                          mode='valid')
```

```python slideshow={"slide_type": "fragment"}
fig, ax = plt.subplots()
ax.plot(step_signal, alpha=0.5, label='signal')
ax.plot(result_conv, linestyle='dashed', label='convolved')
ax.plot(result_corr, linestyle='dashed', label='correlated')
ax.legend(loc='upper left')
ax.margins(y=0.1) 
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
(For technical signal processing reasons, convolutions actually occur "back to front" between the input array and the kernel. Correlations occur in the signal order, so we'll use correlate from now on.)
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "notes"} -->
Whenever neighboring values are close, the filter response is close to 0. Right at the boundary of a step, we're subtracting a small value from a large value and and get a spike in the response. This spike "identifies" our edge.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Commutativity and assortativity of filters
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
What if we try the same trick with our noisy signal?
<!-- #endregion -->

```python
from scipy import ndimage as ndi

noisy_change = ndi.correlate(noisy_signal, np.array([-1, 0, 1]))

fig, ax = plt.subplots()
ax.plot(noisy_signal, label='signal')
ax.plot(noisy_change, linestyle='dashed', label='change')
ax.legend(loc='upper left')
ax.margins(0.1)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
When there is high noise, it becomes much harder to find the spot where the signal changes.

But recall that we smoothed the signal a bit by taking its neighbors. We can apply the two filters in sequence to combine their properties:
<!-- #endregion -->

```python
smooth_change = ndi.correlate(smooth_signal3, np.array([-1, 0, 1]))

fig, ax = plt.subplots()
ax.plot(noisy_signal, alpha=0.5, label='signal')
ax.plot(noisy_change, linestyle='dashed', alpha=0.7, label='change')
ax.plot(smooth_change, label='smooth change')
ax.legend(loc='upper left')
ax.margins(0.1)
ax.hlines([-0.5, 0.5], 0, 100, linewidth=0.5, color='gray');
```

Actually, it turns out that we can do it *in any order* (convolution is *associative*), so we can create a filter that combines both the difference and the mean.

*Note:* we use `np.convolve` here, because it has the option to output a *wider* result than either of the two inputs.

```python editable=true slideshow={"slide_type": ""}
mean_diff = np.correlate([-1, 0, 1], [1/3, 1/3, 1/3], mode='full')

fig, ax = plt.subplots()
ax.plot(mean_diff)
ax.scatter(np.arange(5), mean_diff);
```

We can verify that this gives the same result

```python editable=true slideshow={"slide_type": ""}
smooth_change2 = ndi.correlate(noisy_signal, mean_diff)

fig, ax = plt.subplots()
ax.plot(noisy_signal, alpha=0.5, label='signal')
ax.plot(smooth_change, linestyle='dashed', label='smoothed change')
ax.plot(smooth_change2, label='smoothed change 2')
ax.margins(0.1)
ax.legend(loc='upper left')
ax.hlines([-0.5, 0.5], 0, 100, linewidth=0.5, color='gray');
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## The Gaussian filter
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
The Gaussian kernel with variance $\sigma^2$ is given by:

$$
k_i = \frac{1}{\sqrt{2\pi}\sigma}\exp{\left(-\frac{(x_i - x_0)^2}{2\sigma^2}\right)}
$$

It is an essential smoothing kernel, and has better smoothing properties than the mean kernel, as we'll see in the 2D section below. We create it from scratch here, *and* combine it with a difference kernel, to get an even nicer estimate of our change point:
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
xi = np.arange(19)
x0 = 19 // 2  # 4
x = xi - x0
sigma = 2
kernel = (
    (1 / np.sqrt(np.pi * 2 * sigma**2))
    * np.exp(-x**2 / (2 * sigma**2))
)

diff_kernel = np.convolve(kernel, [-1, 0, 1])
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(12, 4))

ax0.plot(kernel)
ax0.set_title('gaussian kernel')
ax1.plot(diff_kernel)
ax1.set_title('gaussian difference kernel')
ax2.plot(noisy_signal)
ax2.plot(ndi.correlate(noisy_signal, diff_kernel))
ax2.set_title('signal convolved with\ngaussian difference');
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## Local filtering of images
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "notes"} -->
Now let's apply all this knowledge to 2D images instead of a 1D signal. Let's start with an incredibly simple image:
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
import numpy as np

bright_square = np.zeros((7, 7), dtype=float)
bright_square[2:5, 2:5] = 1
```

<!-- #region editable=true slideshow={"slide_type": "notes"} -->
This gives the values below:
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
print(bright_square)
```

<!-- #region slideshow={"slide_type": "notes"} -->
and looks like a white square centered on a black square:
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
fig, ax = plt.subplots()
ax.imshow(bright_square, cmap="gray");
```

<!-- #region slideshow={"slide_type": "slide"} -->
### The mean filter
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
For our first example of a filter, consider the following filtering array, which we'll call a "mean kernel". For each pixel, a kernel defines which neighboring pixels to consider when filtering, and how much to weight those pixels.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
mean_kernel = np.full((3, 3), 1/9)

print(mean_kernel)
```

<!-- #region slideshow={"slide_type": "notes"} -->
Now, let's take our mean kernel and apply it to every pixel of the image.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
Applying a (linear) filter essentially means:
* Center a kernel on a pixel
* Multiply the pixels *under* that kernel by the values *in* the kernel
* Sum all the those results
* Replace the center pixel with the summed result
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
This process is known as convolution.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
Let's take a look at the numerical result:
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
import scipy.ndimage as ndi

%precision 2
print(bright_square)
print(ndi.correlate(bright_square, mean_kernel))
```

<!-- #region slideshow={"slide_type": "notes"} -->
The meaning of "mean kernel" should be clear now: Each pixel was replaced with the mean value within the 3x3 neighborhood of that pixel. When the kernel was over `n` bright pixels, the pixel in the kernel's center was changed to n/9 (= n * 0.111). When no bright pixels were under the kernel, the result was 0.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
This filter is a simple smoothing filter and produces two important results:
1. The intensity of the bright pixel decreased.
2. The intensity of the region near the bright pixel increased.
<!-- #endregion -->

Incidentally, the above filtering is the exact same principle behind the *convolutional neural networks*, or CNNs, that you might have heard much about over the past few years. The only difference is that while above, the simple mean kernel is used, in CNNs, the values inside the kernel are *learned* to find a specific feature, or accomplish a specific task. Time permitting, we'll demonstrate this in an exercise at the end of the notebook.


Here's a small demo of convolution in action.

```python
#--------------------------------------------------------------------------
#  Convolution Demo
#--------------------------------------------------------------------------
from scipy import ndimage as ndi
from matplotlib import patches

def mean_filter_demo(image, vmax=1):
    mean_factor = 1.0 / 9.0  # This assumes a 3x3 kernel.
    iter_kernel_and_subimage = iter_kernel(image)

    image_cache = []

    fig = plt.figure(figsize=(10, 5))
    
    def mean_filter_step(i_step):
        while i_step >= len(image_cache):
            filtered = image if i_step == 0 else image_cache[-1][-1][-1]
            filtered = filtered.copy()

            (i, j), mask, subimage = next(iter_kernel_and_subimage)
            filter_overlay = ski.color.label2rgb(mask, image, bg_label=0,
                                             colors=('cyan', 'red'))
            filtered[i, j] = np.sum(mean_factor * subimage)
            image_cache.append(((i, j), (filter_overlay, filtered)))

        (i, j), images = image_cache[i_step]
        axes = fig.subplots(1, len(images))

        for ax, imc in zip(axes, images):
            ax.imshow(imc, vmax=vmax, cmap="gray")
            rect = patches.Rectangle([j - 0.5, i - 0.5], 1, 1, color='yellow', fill=False)
            ax.add_patch(rect)

        plt.show()
    return mean_filter_step


def mean_filter_interactive_demo(image):
    from ipywidgets import IntSlider, interact
    mean_filter_step = mean_filter_demo(image)
    step_slider = IntSlider(min=0, max=image.size-1, value=0)
    interact(mean_filter_step, i_step=step_slider)


def iter_kernel(image, size=1):
    """ Yield position, kernel mask, and image for each pixel in the image.

    The kernel mask has a 2 at the center pixel and 1 around it. The actual
    width of the kernel is 2*size + 1.
    """
    width = 2*size + 1
    for (i, j), pixel in iter_pixels(image):
        mask = np.zeros(image.shape, dtype='int16')
        mask[i, j] = 1
        mask = ndi.grey_dilation(mask, size=width)
        #mask[i, j] = 2
        subimage = image[bounded_slice((i, j), image.shape[:2], size=size)]
        yield (i, j), mask, subimage


def iter_pixels(image):
    """ Yield pixel position (row, column) and pixel intensity. """
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            yield (i, j), image[i, j]


def bounded_slice(center, xy_max, size=1, i_min=0):
    slices = []
    for i, i_max in zip(center, xy_max):
        slices.append(slice(max(i - size, i_min), min(i + size + 1, i_max)))
    return tuple(slices)
```

```python
mean_filter_interactive_demo(bright_square);
```

<!-- #region slideshow={"slide_type": "notes"} -->
Let's consider a real image now. It'll be easier to see some of the filtering we're doing if we downsample the image a bit. We can slice into the image using the "step" argument to sub-sample it (don't scale images using this method for real work; use `skimage.transform.rescale`):
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
image = ski.data.camera()
pixelated = image[::10, ::10]
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
ax0.imshow(image, cmap="gray")
ax1.imshow(pixelated, cmap="gray") ;
```

<!-- #region slideshow={"slide_type": "notes"} -->
Here we use a step of 10, giving us every tenth column and every tenth row of the original image. You can see the highly pixelated result on the right.
<!-- #endregion -->

We are actually going to be using the pattern of plotting multiple images side by side quite often, so we are going to make the following helper function:

```python
def imshow_all(*images, titles=None):
    images = [ski.util.img_as_float(img) for img in images]

    if titles is None:
        titles = [''] * len(images)
    vmin = min(map(np.min, images))
    vmax = max(map(np.max, images))
    ncols = len(images)
    height = 5
    width = height * len(images)
    fig, axes = plt.subplots(nrows=1, ncols=ncols,
                             figsize=(width, height))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, vmin=vmin, vmax=vmax, cmap="gray")
        ax.set_title(label)
```

<!-- #region slideshow={"slide_type": "slide"} -->
### Mean filter on a real image
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
Now we can apply the filter to this downsampled image:
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
filtered = ndi.correlate(pixelated, mean_kernel)
imshow_all(pixelated, filtered, titles=['pixelated', 'mean filtered'])
```

<!-- #region slideshow={"slide_type": "notes"} -->
Comparing the filtered image to the pixelated image, we can see that this filtered result is smoother: Sharp edges (which are just borders between dark and bright pixels) are smoothed because dark pixels reduce the intensity of neighboring pixels and bright pixels do the opposite.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Essential filters
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
If you read through the last section, you're already familiar with the essential concepts of image filtering. But, of course, you don't have to create custom filter kernels for all of your filtering needs. There are many standard filter kernels pre-defined from half a century of image and signal processing.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "fragment"} -->
### Gaussian filter
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
The classic image filter is the Gaussian filter. This is similar to the mean filter, in that it tends to smooth images. The Gaussian filter, however, doesn't weight all values in the neighborhood equally. Instead, pixels closer to the center are weighted more than those farther away.
<!-- #endregion -->

```python slideshow={"slide_type": "notes"}
# Rename module so we don't shadow the builtin function
smooth_mean = ndi.correlate(bright_square, mean_kernel)
sigma = 1
smooth = ski.filters.gaussian(bright_square, sigma)
imshow_all(bright_square, smooth_mean, smooth,
           titles=['original', 'result of mean filter', 'result of gaussian filter'])
```

<!-- #region slideshow={"slide_type": "notes"} -->
For the Gaussian filter, `sigma`, the standard deviation, defines the size of the neighborhood.

For a real image, we get the following:
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
# The Gaussian filter returns a float image, regardless of input.
# Cast to float so the images have comparable intensity ranges.
pixelated_float = ski.util.img_as_float(pixelated)
smooth = ski.filters.gaussian(pixelated_float, sigma=1)
imshow_all(pixelated_float, smooth)
```

<!-- #region editable=true slideshow={"slide_type": "notes"} -->
This doesn't look drastically different than the mean filter, but the Gaussian filter is typically preferred because of the distance-dependent weighting, and because it does not have any sharp transitions (consider what happens in the Fourier domain!). For a more detailed image and a larger filter, you can see artifacts in the mean filter since it doesn't take distance into account:
<!-- #endregion -->

```python editable=true slideshow={"slide_type": "fragment"}
size = 20
structuring_element = np.ones((3*size, 3*size))
smooth_mean = ski.filters.rank.mean(image, structuring_element)
smooth_gaussian = ski.filters.gaussian(image, size)
titles = ['mean', 'gaussian']
imshow_all(smooth_mean, smooth_gaussian, titles=titles)
```

<!-- #region slideshow={"slide_type": "notes"} -->
(Above, we've tweaked the size of the structuring element used for the mean filter and the standard deviation of the Gaussian filter to produce an approximately equal amount of smoothing in the two results.)
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
Incidentally, for reference, let's have a look at what the Gaussian filter actually looks like. Technically, the value of the kernel at a pixel that is $r$ rows and $c$ cols from the center is:

$$
k_{r, c} = \frac{1}{2\pi \sigma^2} \exp{\left(-\frac{r^2 + c^2}{2\sigma^2}\right)}
$$

Practically speaking, this value is pretty close to zero for values more than $4\sigma$ away from the center, so practical Gaussian filters are truncated at about $4\sigma$:
<!-- #endregion -->

```python
sidelen = 45
sigma = (sidelen - 1) // 2 // 4
spot = np.zeros((sidelen, sidelen), dtype=float)
spot[sidelen // 2, sidelen // 2] = 1
kernel = ski.filters.gaussian(spot, sigma=sigma)

imshow_all(spot, kernel / np.max(kernel))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
If we have a look at a slice of the filtered result at its midpoint, we see the classic shape of a Gaussian kernel in one dimension:
<!-- #endregion -->

```python
fig, ax = plt.subplots(1, 2, figsize=[9, 3], width_ratios=[1, 1.5])

ax[0].imshow(kernel, cmap='gray')
ax[0].vlines(kernel.shape[1] // 2, -100, 100)
ax[0].set_ylim((sidelen - 1, 0))

ax[1].plot(kernel[:, kernel.shape[1] // 2])
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
## Basic edge filtering
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "notes"} -->
For images, edges are boundaries between light and dark values. The detection of edges can be useful on its own, or it can be used as preliminary step in other algorithms (which we'll see later).
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Difference filters in 2D
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
For images, you can think of an edge as points where the gradient is large in one direction. We can approximate gradients with difference filters.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
vertical_kernel = np.array([
    [-1],
    [ 0],
    [ 1],
])

gradient_vertical = ndi.correlate(pixelated.astype(float),
                                  vertical_kernel)
fig, ax = plt.subplots()
ax.imshow(gradient_vertical, cmap="gray");
```

```python
vertical_kernel.shape
```

<!-- #region slideshow={"slide_type": "fragment"} -->
### Exercise 3:
<!-- #endregion -->

- Add a horizontal kernel to the above example to also compute the horizontal gradient, $g_y$
- Compute the magnitude of the image gradient at each point: $\left|g\right| = \sqrt{g_x^2 + g_y^2}$

```python
# Solution here
```

```python
# Solution here
```

<!-- #region editable=true slideshow={"slide_type": "slide"} -->
### Sobel edge filter
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
The Sobel filter, the most commonly used edge filter, should look pretty similar to what you developed above. Take a look at the vertical and horizontal components of the Sobel kernel to see how they differ from your earlier implementation:

* http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.sobel_v
* http://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.sobel_h
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
imshow_all(bright_square, ski.filters.sobel(bright_square))
```

<!-- #region slideshow={"slide_type": "notes"} -->
Notice that the size of the output matches the input, and the edges aren't preferentially shifted to a corner of the image. Furthermore, the weights used in the Sobel filter produce diagonal edges with reponses that are comparable to horizontal or vertical edges.

Like any derivative, noise can have a strong impact on the result:
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
pixelated_gradient = ski.filters.sobel(pixelated)
imshow_all(pixelated, pixelated_gradient)
```

<!-- #region slideshow={"slide_type": "notes"} -->
Smoothing is often used as a preprocessing step in preparation for feature detection and image-enhancement operations because sharp features can distort results.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
gradient = ski.filters.sobel(smooth)
titles = ['gradient before smoothing', 'gradient after smoothing']
# Scale smoothed gradient up so they're of comparable brightness.

imshow_all(pixelated_gradient, gradient*1.8, titles=titles)
```

<!-- #region slideshow={"slide_type": "notes"} -->
Notice how the edges look more continuous in the smoothed image.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "fragment"} -->
## Nonlinear filters
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": "notes"} -->
At this point, we make a distinction. The earlier filters were implemented as a *linear dot-product* of values in the filter kernel and values in the image. The following kernels implement an *arbitrary* function of the local image neighborhood. Denoising filters in particular are filters that preserve the sharpness of edges in the image.

As you can see from our earlier examples, mean and Gaussian filters smooth an image rather uniformly, including the edges of objects in an image. When denoising, however, you typically want to preserve features and just remove noise. The distinction between noise and features can, of course, be highly situation-dependent and subjective.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "notes"} -->
The median filter is the classic edge-preserving filter. As the name implies, this filter takes a set of pixels (i.e. the pixels within a kernel or "structuring element") and returns the median value within that neighborhood. Because regions near a sharp edge will have many dark values and many light values (but few values in between) the median at an edge will most likely be either light or dark, rather than some value in between. In that way, we don't end up with edges that are smoothed.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
neighborhood = ski.morphology.disk(radius=1)  # "selem" is often the name used for "structuring element"
median = ski.filters.rank.median(pixelated, neighborhood)
titles = ['image', 'gaussian', 'median']
imshow_all(pixelated, smooth, median, titles=titles)
```

<!-- #region slideshow={"slide_type": "notes"} -->
This difference is more noticeable with a more detailed image.
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
neighborhood = ski.morphology.disk(10)
coins = ski.data.coins()
mean_coin = ski.filters.rank.mean(coins, neighborhood)
median_coin = ski.filters.rank.median(coins, neighborhood)
titles = ['image', 'mean', 'median']
imshow_all(coins, mean_coin, median_coin, titles=titles)
```

<!-- #region slideshow={"slide_type": "notes"} -->
Notice how the edges of coins are preserved after using the median filter.
<!-- #endregion -->

## Further reading

See the scikit-image [filters API documentation](https://scikit-image.org/docs/dev/api/skimage.filters.html) for further reading. The scikit-image [restoration module](https://scikit-image.org/docs/dev/api/skimage.restoration.html) also includes sophisticated modules for image denoising and deconvolution.
