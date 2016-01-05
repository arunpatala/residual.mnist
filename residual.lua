require 'nn'
require 'cunn'

local function convunit(net,fin,fout,fsize,str,pad,nobatch)
    local nobatch = nobatch or false
    local pad = pad or 1
    local str = str or 1
    local fsize = fsize or 3
    net:add(nn.SpatialConvolution(fin,fout,fsize,fsize,str,str,pad,pad))
    if(nobatch==false) then net:add(nn.SpatialBatchNormalization(fout)) end
    net:add(nn.ReLU(true))
end
local function convunit31(net,fin,half,str,nobatch)
    local str = str or 3
    local half = half or false
    if(half) then
        convunit(net,fin,2*fin,3,2,nil,nobatch)
    else convunit(net,fin,fin,3,1,nil,nobatch) end
end
local function convunit2(net,fin,half)
    local half = half or false
    convunit31(net,fin,half,nil,true)
    if(half) then convunit31(net,2*fin) 
    else convunit31(net,fin) end
end

local function resUnit(net, unit, fin, half)
    local half = half or false
    local net = net or nn.Sequential()
    local cat = nn.ConcatTable()
    cat:add(unit)
    if(half==false) then
        cat:add(nn.Identity())
    else 
        cat:add(nn.SpatialConvolution(fin,2*fin,1,1,2,2))
    end
    net:add(cat)
    net:add(nn.CAddTable())
    net:add(nn.ReLU(true))
    return net
end
local function rconvunit2(net,fin,half)
    local unit = nn.Sequential()
    convunit2(unit,fin,half)
    resUnit(net,unit,fin,half)
end

local function rconvunitN(net,fin,N)
    local N = N or 0
    for i=1,N do
        rconvunit2(net,fin)
    end
end

local res = {}
res.convunit = convunit
res.rconvunit2 = rconvunit2
res.rconvunitN = rconvunitN
return res