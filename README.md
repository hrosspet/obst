# Obstacle Tower Challenge

This project serves for the Unity challenge Obstacle Tower. It offers simple environments for easy testing of hypotheses leading towards the solution of the challenge.

# Development

0. `git clone https://github.com/hrosspet/obst.git`
1. `cd obst`
2. `pip install -r requirements.txt`
3. `pip install -e .`

# Execution

You can run experiments from command line `python obst/__main__.py` with optional `DEBUG` param to see more info in the log. Or use as a lib in jupyter / google colab.

# Using Docker

The project can be used as a docker image in order to be portable

Download latest [Tensorflow Docker image](https://www.tensorflow.org/install/docker).

	docker pull tensorflow/tensorflow:latest-py3

Create the image defined by the Dockerfile.
This builds on the TensorFlow image and also does the installation that was described earlier.
The current state of our source code is copied into our image.

	docker build -t obstapp .

To run:

	docker run obstapp

This is like doing `python3 obst/__main__.py`.

## Development

To make changes, edit the files *outside of the image*. To then run them inside the image, do:

	docker run -v /home/albert/github/obst:/obst obstapp

This mounts our current source directory instead of the source code that was copied.
To update the source code inside the image, rebuild it.

