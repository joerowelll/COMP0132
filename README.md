# COMP0132 Robotics and Computation Project
## Semantic Validation in Structure From Motion

## Table of Contents
### Structure From Motion
### Semantic Segmentation
### Datasets
### Misc


### Semantic Segmentation on Brighton Data-set Demo
https://www.youtube.com/watch?v=UwfRyR7IwWU&t=55s



### [COLMAP](https://colmap.github.io/) Dependencies
```
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev
```
Install Ceres Solver
```
sudo apt-get install libatlas-base-dev libsuitesparse-dev
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout 1.14.0 # do not use the latest release
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
make -j
sudo make install
```
Configure and Compile COLMAP
```
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout dev
mkdir build
cd build
cmake ..
make -j
sudo make install
```

### [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab) Dependencies
```
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
sudo apt-get install python-pil python-numpy
pip install --user jupyter
pip install --user matplotlib
pip install --user PrettyTable
```
### Database Manipulation Dependencies
```
sudo apt-get update
sudo apt-get install python-argparse \
	sqlite3 \
	numpy

```
