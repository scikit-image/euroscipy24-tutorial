# scikit-image tutorial at EuroSciPy 2024

This repository contains the teaching materials for the scikit-image tutorial
at EuroSciPy 2024.


## Development

### Run notebooks in local environment

```shell
python -m pip install -r requirements/local_dev.txt
```

If you want to add a dependency to `local_dev.txt`, add it to the corresponding `local_dev.in` instead and run

```shell
uv pip compile -o requirements/local_dev.txt requirements/local_dev.in
```


### Build & serve website locally

Setup the required dependencies with

```shell
python -m pip install -r requirements/jupyterlite.txt
```

and then run

```shell
jupyter lite build --content tutorial/
jupyter lite serve
```

If you want to add a dependency to `jupyterlite.txt`, add it to the corresponding `jupyterlite.in` instead and run

```shell
uv pip compile -o requirements/jupyterlite.txt requirements/jupyterlite.in
```


## Acknowledgements

Builds on the original [scikit-image tutorials](https://github.com/scikit-image/skimage-tutorials), last years tutorial at [EuroSciPy 2023](https://github.com/glemaitre/euroscipy-2023-scikit-image) by Guillaume Lemaitre, and the [joined image analysis tutorial at SciPy 2024](https://github.com/scipy-2024-image-analysis/tutorial).
