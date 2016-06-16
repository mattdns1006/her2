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
		:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0))
		:add(SBatchNorm(nOutputPlane))
end
	
local function basicblock(nInputPlane, n, stride)
	local s = nn.Sequential()

	s:add(Convolution(nInputPlane,n,3,3,1,1,1,1))
	s:add(SBatchNorm(n))
	s:add(ReLU(true))
	--[[
	s:add(Convolution(n,n,2,2,1,1,1,1))
	s:add(SBatchNorm(n))
	s:add(ReLU(true))
	]]--
	s:add(Max(3,3,stride,stride,1,1))
	--s:add(Convolution(n,n,3,3,2,2,1,1))
	--s:add(SBatchNorm(n))
	--s:add(Max(3,3,stride,stride,1,1))


	return nn.Sequential()
	 :add(nn.ConcatTable()
	    :add(s)
	    :add(shortcut(nInputPlane, n, stride)))
	 :add(nn.CAddTable(true))
	 :add(ReLU(true))
end

models = {}

function models.nonResNet()

	local model 
	local nLayers = math.ceil(math.log(params.windowSize,2)) 
	local function miniNet(nWindows)
		local model = nn.Sequential()
		local nFeats = params.nFeats
		local nInputs
		for i = 1, nLayers do
			if i == 1 then nInputs = 3; else nInputs = nFeats*(i-1); end
			print(nFeats*i)
			model:add(Convolution(nInputs,nFeats*i,3,3,1,1,1,1))
			model:add(SBatchNorm(nFeats*i))
			model:add(ReLU(true))
			model:add(nn.SpatialMaxPooling(3,3,2,2,1,1))
			--model:add(Convolution(nFeats*i,nFeats*i,3,3,2,2,1,1))
			--model:add(SBatchNorm(nFeats*i))
			--model:add(ReLU(true))

		end
		local nOutputs = model:forward(torch.rand(nWindows,3,params.windowSize,params.windowSize)):size()
		local nOutputsProd = nOutputs[1]*nOutputs[2]*nOutputs[3]*nOutputs[4]
		model:add(nn.View())
		model:add(nn.Linear(nOutputsProd,13))
		model:add(nn.Sigmoid())
		layers.init(model)
		return model
	end

	model = miniNet(params.nHER2Windows)
	layers.init(model)
	return model 
end


function models.resNet()

	local model 
	local nLayers = math.ceil(math.log(params.windowSize,2)) 
	local function miniNet(nWindows)
		local model = nn.Sequential()
		local nFeats = params.nFeats
		local nIn = 32
		--model:add(Convolution(3,nIn,3,3,1,1,1,1))
		model:add(SBatchNorm(3))
		local nInputs
		for i = 1, nLayers do
			if i == 1 then nInputs = 3; else nInputs = nFeats; end
			model:add(basicblock(nInputs,nFeats,2))
		end
		local nOutputs = model:forward(torch.rand(nWindows,3,params.windowSize,params.windowSize)):size()
		print(nOutputs)
		local nOutputsProd = nOutputs[1]*nOutputs[2]*nOutputs[3]*nOutputs[4]
		model:add(nn.View())
		model:add(nn.Linear(nOutputsProd,25))
		model:add(ReLU(true))
		model:add(nn.Linear(25,13))
		model:add(nn.Sigmoid())
		layers.init(model)
		return model
	end

	model = miniNet(params.nHER2Windows)
	layers.init(model)
	return model 
end

function models.resNetSiamese()

	local model = nn.Sequential()

	local paraNet = nn.ParallelTable()	
	
	local nLayers = math.ceil(math.log(params.windowSize,2)) - 1
	local function miniNet(nWindows)
		local model = nn.Sequential()
		local nFeats = params.nFeats
		--model:add(Convolution(3,22,3,3,1,1,1,1))
		--model:add(SBatchNorm(22))
		local nInputs
		for i = 1, nLayers do
			if i == 1 then nInputs = 3; else nInputs = nFeats; end
			model:add(basicblock(nInputs,nFeats,2))
		end
		model:add(nn.View(nWindows*nFeats*4,1))
		model:add(Convolution1D(1,10,1,1))
		model:add(nn.BatchNormalization(10))
		model:add(ReLU(true))
		model:add(Max1D(3,2))
		model:add(Convolution1D(10,1,1,1))
		model:add(ReLU(true))
		local nOutputs = model:forward(torch.rand(nWindows,3,params.windowSize,params.windowSize)):size(1)
		model:add(nn.View(1,nOutputs))
		model:add(nn.Linear(nOutputs,25))
		model:add(ReLU(true))
		return model
	end

	paraNet:add(miniNet(params.nHER2Windows)):add(miniNet(params.nHEWindows))
	layers.init(paraNet)
	model:add(paraNet):add(nn.JoinTable(2))
	model:add(nn.Linear(50,25))
	model:add(ReLU(true))
	model:add(nn.Linear(25,13))
	model:add(nn.Sigmoid())
	layers.init(model)
	model:float()
	return model 
end
--[[
params = {}
params.windowSize = 256 
params.nHER2Windows = 5 
params.nHEWindows = 5 
params.nFeats = 24
local x = torch.rand(params.nHER2Windows,3,params.windowSize,params.windowSize):cuda()
local x = torch.rand(params.nHEWindows,3,params.windowSize,params.windowSize):cuda()
local X = {x,x}
m = models.nonResNet():cuda()
print(m:forward(x))
]]--
return models

