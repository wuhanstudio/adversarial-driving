## Adversarial Driving

> Attacking End-to-End Autonomous Driving

[[ Talk ]](https://driving.wuhanstudio.uk) [[ Video ]](https://youtu.be/I0i8uN2oOP0) [[ Paper ]](https://arxiv.org/abs/2103.09151) [[ Code ]](https://github.com/wuhanstudio/adversarial-driving)

The behaviour of end-to-end autonomous driving model can be manipulated by adding unperceivable perturbations to the input image.

[![](./doc/adversarial-driving.png)](https://driving.wuhanstudio.uk)

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

This simulator was built for Udacity's Self-Driving Car Nanodegree, and it's available [here](https://github.com/udacity/self-driving-car-sim) (Download the zip file, extract it and run the executable file).

```
$ cd adversarial-driving/simulator/
$ ./Default\ Linux\ desktop\ Universal.x86_64
```

Version 1, 12/09/16

[Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)
[Mac](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip)
[Windows 32](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip)
[Windows 64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip)


Your can use any web server, just serve all the content under **client/web**.

If you use windows, click on **client/client.exe**. It's a single executable that packages everything.

For Linux and Mac, or other Unix, the server can be built with:

```
$ cd adversarial-driving/client
$ go install github.com/gobuffalo/packr/v2@v2.8.3
$ go build
$ ./client
```

The web page is available at: http://localhost:3333/

<!-- <a href="https://youtu.be/DOdaiGxgHEs"><img src="./doc/video.png" /></a> -->

## Adversarial ROS Driving

We also tested our attacks in ROS Gazebo simulator. 

https://github.com/wuhanstudio/adversarial-ros-driving

[![](https://raw.githubusercontent.com/wuhanstudio/adversarial-ros-driving/master/doc/adversarial-ros-driving.png)](https://github.com/wuhanstudio/adversarial-ros-driving)

## Citation

```
@misc{han2021driving,
  doi = {10.48550/ARXIV.2103.09151},
  url = {https://arxiv.org/abs/2103.09151},
  author = {Wu, Han and Yunas, Syed and Rowlands, Sareh and Ruan, Wenjie and Wahlstrom, Johan},
  title = {Adversarial Driving: Attacking End-to-End Autonomous Driving},
  publisher = {arXiv},
  year = {2021}
}
```
