local nn = require 'nn'
require 'cunn'
layers = require "layers"

local Convolution = nn.SpatialConvolution
local Convolution1D = nn.TemporalConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local Max1D = nn.TemporalMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function shortcut(nInputPlane, nOutputPlane, stride)
	return nn.Sequential()
		:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride))
		:add(SBatchNorm(nOutputPlane))
end
	
local function basicblock(nInputPlane, n, stride)
	local s = nn.Sequential()
	s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
	s:add(SBatchNorm(n))
	s:add(ReLU(true))
	s:add(Convolution(n,n,3,3,1,1,1,1))
	s:add(SBatchNorm(n))

	return nn.Sequential()
	 :add(nn.ConcatTable()
	    :add(s)
	    :add(shortcut(nInputPlane, n, stride)))
	 :add(nn.CAddTable(true))
	 :add(ReLU(true))
end

models = {}

function models.resNetSiamese()

	local model = nn.Sequential()

	local paraNet = nn.ParallelTable()	
	
	local nLayers = math.ceil(math.log(params.windowSize,2)) - 1
	local function miniNet()
		local model = nn.Sequential()
		local nFeats = params.nFeats
		local nInputs
		for i = 1, nLayers do
			if i == 1 then nInputs = 3; else nInputs = nFeats; end
			model:add(basicblock(nInputs,nFeats,2))
		end
		model:add(nn.View(params.nWindows*nFeats*4,1))
		model:add(Convolution1D(1,10,1,1))
		model:add(nn.BatchNormalization(10))
		model:add(ReLU(true))
		model:add(Max1D(3,2))
		model:add(Convolution1D(10,1,1,1))
		model:add(ReLU(true))
		local nOutputs = model:forward(torch.rand(params.nWindows,3,params.windowSize,params.windowSize)):size(1)
		model:add(nn.View(1,nOutputs))
		model:add(nn.Linear(nOutputs,25))
		model:add(ReLU(true))
		return model
	end
	layers.init(paraNet)
	paraNet:add(miniNet()):add(miniNet())
	model:add(paraNet):add(nn.JoinTable(2))
	model:add(nn.Linear(50,25))
	model:add(ReLU(true))
	model:add(nn.Linear(25,2))
	return model 
end

params = {}
params.windowSize = 112 
params.nWindows = 5 
params.nFeats = 24
x = torch.rand(params.nWindows,3,params.windowSize,params.windowSize):cuda()
X = {x,x}
m = models.resNetSiamese():cuda()
print(m:forward(X))

return models

