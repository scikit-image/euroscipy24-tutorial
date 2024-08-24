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

# scikit-image tutorial at EuroSciPy 2024

**Scientists are producing more and more images with telescopes, microscopes, MRI scanners, etc. They need automatable tools to measure what they've imaged and help them turn these images into knowledge. This tutorial covers the fundamentals of algorithmic image analysis, starting with how to think of images as NumPy arrays, moving on to basic image filtering, and finishing with a complete workflow: segmenting a 3D image into regions and making measurements on those regions.**

This web app contains the teaching materials for the [scikit-image tutorial at EuroSciPy 2024](https://pretalx.com/euroscipy-2024/talk/ZVBAKK/).
It's a JupyterLite application that uses Pyodide to run Python directly in your browser! You don't need to create a local Python environment and can directly start.

The tutorial is aimed at folks who have some experience in scientific computing with Python, but are new to image analysis. We will introduce the fundamentals of working with images in scientific Python. At every step, we will visualize and understand our work using Matplotlib.

- [Part 1: Images are just NumPy arrays](01_images_are_arrays.ipynb). In this section we will cover the basics: how to think of images not as things we can see but numbers we can analyze.
- [Part 2: Changing the structure of images with image filtering](02_image_filtering.ipynb). In this section we will define filtering, a fundamental operation on signals (1D), images (2D), and higher-dimensional images (3D+). We will use filtering to find various structures in images, such as blobs and edges.
