# residual.mnist

Prereq: Install torch and nn packages. Also CUDA with GPU is required as the computations are huge.
For MNIST dataset, install https://github.com/andresy/mnist

Training 100+ layers neural network using residual networks attaining
99.5% accuracy without data augmentation or dropout or ensembles

Usage: th main.lua --batchSize 128 --layers 100

Paper: http://arxiv.org/abs/1512.03385

Blog: https://deepmlblog.wordpress.com/2016/01/05/residual-networks-in-torch-mnist/


