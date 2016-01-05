require 'nn';
require 'cunn';

local cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST training using Residual Neural Networks')
cmd:text('Example:')
cmd:text('$> th main.lua --layers 100 --batchSize 128 --iterations 10')
cmd:text('Options:')
cmd:option('--momentum', 0.9, 'momemtum during SGD')
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--learningRateDecay', 5.0e-6, 'learning rate decay')
cmd:option('--iterations', 30, 'number of iterations to run')
cmd:option('--batchSize', 128, 'batch size(adjust to fit in GPU)')
cmd:option('--layers', 100, 'approx num of layers to train')


cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
   print(opt)
end

opt.momentum = tonumber(opt.momentum)
opt.learningRate = tonumber(opt.learningRate)
opt.learningRateDecay = tonumber(opt.learningRateDecay)
opt.iterations = tonumber(opt.iterations)
opt.batchSize = tonumber(opt.batchSize)
opt.layers = tonumber(opt.layers)

local N = (opt.layers-10)/6


local mnist = require 'mnist'
local train = mnist.traindataset()
local Xt = train.data
local Yt = train.label
local test = mnist.testdataset()
local Xv = test.data
local Yv = test.label
Yt[Yt:eq(0)] = 10
Yv[Yv:eq(0)] = 10
local train = require 'train'
local model = require 'model'
local net,ct = model.residual(N)
print(net:__tostring__())

local sgd_config = {
      learningRate = opt.learningRate,
      learningRateDecay = opt.learningRateDecay,
      momentum = opt.momemtum
   }
print('Number of convolutional layers .. '..#net:findModules('nn.SpatialConvolution'))
train.sgd(net,ct,Xt,Yt,Xv,Yv,opt.iterations,sgd_config,opt.batchSize)


