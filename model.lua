require 'nn'

local model = {}
local res = require 'residual'


function model.residual(N)
    local N = N or 15
    local half = true

    net = nn.Sequential()
    net:add(nn.Reshape(1,28,28))
    res.convunit(net,1,64)
    res.rconvunitN(net,64,N)
    res.rconvunit2(net,64,half)
    res.rconvunitN(net,128,N)
    res.rconvunit2(net,128,half)
    res.rconvunitN(net,256,N)
    res.rconvunit2(net,256,half)

    cls = nn.Sequential()
    local wid = 4
    cls:add(nn.Reshape(512*wid*wid))
    cls:add(nn.Linear(512*wid*wid,10))
    cls:add(nn.LogSoftMax())
    net:add(cls)
    local ct = nn.ClassNLLCriterion()

    require 'cunn';
    net = net:cuda()
    ct = ct:cuda()

    return net,ct
end
return model