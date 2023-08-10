# CCN Tutorial - A high-dimensional view of computational neuroscience

Welcome!

This repository contains material for a [tutorial](https://2023.ccneuro.org/kt3.php) presented at [Cognitive Computational Neuroscience 2023](https://2023.ccneuro.org/).

## Instructions

The [Jupyter notebooks](https://docs.jupyter.org/en/latest/index.html) for this tutorial are designed to be run on [Google Colab](https://colab.research.google.com/):

- TODO add direct links to Colab with the notebooks

If you'd just like to follow along, you can view the notebooks rendered on [our website](https://bonnerlab.github.io/ccn-tutorial/).

If you'd prefer to work locally, you can set up a local environment to run these notebooks:

- Create a [Python virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment) with `Python >=3.10.12`.
- Install this package to ensure all the required dependencies are present: `pip install git+https://github.com/BonnerLab/ccn-tutorial.git`.
- Clone the repository locally: `git clone --depth=1 https://github.com/BonnerLab/ccn-tutorial`
- Access the Jupyter notebooks (`.ipynb` files) at `ccn-tutorial/docs/pages`

## TODO

- [ ] homepage
  - [ ] change motivation
  - [ ] fix human metadata TODOs
- [ ] PCA notebook
  - [ ] Fix title, subtitle, summary
  - [ ] Fix Google Colab link
  - [ ] add animation for geometric intuition
  - [ ] display data projected onto PCs as well
  - [ ] increase number of neurons - show spectrum with clean break at 2 dims
- [ ] NSD notebook
  - [ ] replace data with nsdgeneral
  - [ ] stress-test server with heavy downloads
- [ ] PLS-SVD notebook