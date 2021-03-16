####  2. Setup the server

You may use [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html). 

```python
$ cd model
$ conda env create -f environment.yml
$ conda activate adversarial-driving
$ python drive.py model.h5
```