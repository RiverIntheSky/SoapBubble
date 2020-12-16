# Introduction
This is the code to my paper [Chemomechanical simulation of soap film flow on spherical bubbles](https://doi.org/10.1145/3386569.3392094)
> :warning: only tested on an arch linux machine with GTX 1080

# Quickstart

## Requires
* [AmgX](https://developer.nvidia.com/amgx)
* CUDA
* Boost
* OpenCV

## Build

```
git clone https://github.com/RiverIntheSky/SoapBubble.git
cd SoapBubble
mkdir build
cd build
cmake ..
make
```

## Usage

```
./soapBubble ../config.txt
```
Please have a look at the config file `config.txt`, and the physical meanings of the quantities in `include/bubble.cuh`.

## Setting air flow

The air flow velocity can be initialized via a custom texture as in `init/velocityfield_512.exr`. Velocity fields are scaled so that maximum speed is 1.0. Channels are u_theta, u_phi, 0. Additionally, one can hard-code a time-varying wind velocity field in `airFlowU()` and `airFlowV()`.

## Setting other things
Please have a look at those `# define`s in `include/header.cuh` and how they are used in `kernel/bubble.cu`.


## Output
A sequence of grayscale textures storing the thickness of the film, scaled by H in config.txt

## Side notes
Attempts on bubbles breaking and Van der Waals forces is made, with little success.
