layers = require "layers"

models = {}

local function ConvInit(name,block)
	for k,v in pairs(block:findModules(name)) do
		local n = v.kW*v.kH*v.nOutputPlane
		v.weight:normal(0,math.sqrt(2/n))
		if cudnn.version >= 4000 then
		v.bias = nil
		v.gradBias = nil
		else
		v.bias:zero()
		end
	end
end

local function BNInit(name,block)
	for k,v in pairs(block:findModules(name)) do
		v.weight:fill(1)
		v.bias:zero()
	end
end

local function linearInit(name)
	for k,v in pairs(model:findModules('nn.Linear')) do
	      v.bias:zero()
	end
end

function models.model1()

	 nFeats = params.nFeats 
	 nLayers = params.nLayers 
	 cnn_filters = torch.range(nFeats,nFeats*nLayers,nFeats)
	 cnn_filters:fill(nFeats)
	 cnn_filter_size = torch.Tensor(nLayers):fill(2)
	 cnn_stride = torch.Tensor(nLayers):fill(1)
	 cnn_padding = torch.Tensor(nLayers):fill(1)
	
	 mp_filter_size = torch.Tensor(nLayers):fill(3)
	 mp_stride = torch.Tensor(nLayers):fill(2)
	 mp_padding = torch.Tensor(nLayers):fill(0)
	
	model = nn.Sequential()
	
	j = 0
	--model:add(nn.SpatialBatchNormalization(3))
	for i = 1, nLayers -1 do
		j = j + 1
		layers.add_cnn(model,i)
		model:add(nn.SpatialBatchNormalization(cnn_filters[i]))
		layers.add_af(model)
		layers.add_mp(model,i)

	end

	layers.add_cnn(model,cnn_filters:size(1))
	model:add(nn.SpatialBatchNormalization(cnn_filters[{-1}]))
	layers.add_af(model)

	local outputSize = model:forward(torch.randn(params.nWindows,3,params.windowSize,params.windowSize)):size()
	print("==>Output size before reshape")
	print(outputSize)
	local nOutputsDense = outputSize[2]*outputSize[3]*outputSize[4] 
	local feats = 50

	splitter = nn.ConcatTable()
	all,  oneByOne = nn.Sequential(),nn.Sequential()
	local function findAf(model) return model:add(nn.Sigmoid()) end

	-- All
	all:add(nn.View(nOutputsDense*params.nWindows))
	all:add(nn.Linear(nOutputsDense*params.nWindows,feats))
	layers.add_af(all)
	all:add(nn.Linear(feats,feats))
	layers.add_af(all)
	all:add(nn.Linear(feats,1))
	--findAf(oneByOne)

	-- One by one channel
	oneByOne:add(nn.View(nOutputsDense))
	--oneByOne:add(nn.BatchNormalization(nOutputsDense))
	oneByOne:add(nn.Linear(nOutputsDense,1))
	--findAf(oneByOne)


	local blocks = {oneByOne,all,model}

	local function init(block)
		ConvInit('cudnn.SpatialConvolution',block)
		ConvInit('nn.SpatialConvolution',block)
		BNInit('fbnn.SpatialBatchNormalization',block)
		BNInit('cudnn.SpatialBatchNormalization',block)
		BNInit('nn.SpatialBatchNormalization',block)
		for k,v in pairs(block:findModules('nn.Linear')) do
		      v.bias:zero()
		end
	end
	init(oneByOne)
	init(all)
	init(model)

	splitter:add(all)
	splitter:add(oneByOne)
	model:add(splitter)
	model:add(nn.JoinTable(1))
	--model:add(nn.Sigmoid())

	return model 
end


function models.main()
	require "cunn"
	params = {} 
	params.windowSize = 512 
	params.nFeats = 16
	params.nWindows = 10 
	model = models.model1():cuda()
	print(model)
	X = torch.randn(params.nWindows,3,params.windowSize,params.windowSize):cuda()
	print(X:size())

	print(model:forward(X))
end

return models



		
	


	
