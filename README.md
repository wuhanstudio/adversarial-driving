## Adversarial Driving

> Attacking End-to-End Autonomous Driving

[Talk](https://driving.wuhanstudio.uk) [Video] [Paper](https://arxiv.org/abs/2103.09151) [Code](https://github.com/wuhanstudio/adversarial-driving)


### Quick Start

You may use [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html). 

```python
$ git clone https://github.com/wuhanstudio/adversarial-driving
$ cd adversarial-driving/model

$ # CPU
$ conda env create -f environment.yml
$ conda activate adversarial-driving

$ # GPU
$ conda env create -f environment_gpu.yml
$ conda activate adversarial-gpu-driving

$ python drive.py model.h5
```

The web page will be available at: http://localhost:9090/

This simulator was built for Udacity's Self-Driving Car Nanodegree, and it's available [here](https://github.com/udacity/self-driving-car-sim).

Instruction: Download the zip file, extract it and run the executable file.

Version 1, 12/09/16

[Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
[Mac](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
[Windows 32](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
[Windows 64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)


### Overview

The behaviour of end-to-end autonomous driving model can be manipulated by adding unperceivable perturbations to the input image.

<a href="https://youtu.be/DOdaiGxgHEs"><img src="./doc/video.png" /></a>

![](./doc/adversarial-driving.png)

```
@misc{wu2021adversarial,
      title={Adversarial Driving: Attacking End-to-End Autonomous Driving Systems}, 
      author={Han Wu, Syed Yunas and Wenjie Ruan},
      year={2021},
      eprint={2103.09151},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
