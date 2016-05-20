layers = require "layers"

models = {}

function models.ConvInit(name,block)
	for k,v in pairs(block:findModules(name)) do
		local n = v.kW*v.kH*v.nOutputPlane
		v.weight:normal(0,math.sqrt(2/n))
		--[[
		if cudnn.version >= 4000 then
		v.bias = nil
		v.gradBias = nil
		else
		v.bias:zero()
		]]--
		v.bias:zero()
	end
end

function models.BNInit(name,block)
	for k,v in pairs(block:findModules(name)) do
		v.weight:fill(1)
		v.bias:zero()
	end
end

function models.linearInit(block)
	for k,v in pairs(block:findModules('nn.Linear')) do
	      v.bias:zero()
	end
end

function models.init(block)
	models.ConvInit('cudnn.SpatialConvolution',block)
	models.ConvInit('nn.SpatialConvolution',block)
	models.BNInit('fbnn.SpatialBatchNormalization',block)
	models.BNInit('cudnn.SpatialBatchNormalization',block)
	models.BNInit('nn.SpatialBatchNormalization',block)
	models.linearInit(block)
	--[[
	for k,v in pairs(block:findModules('nn.Linear')) do
	      v.bias:zero()
	end
	]]--
end

local function ShareGradInput(module,key)
	assert(key)
	module.__shareGradInputKey = key
	return module
end

local function shortcut(type)
	if type == "C" then
		return nn.Sequential()
	end
	return nn.Identity()
end

function residualBlock(model,i)
	local resBlock = nn.Sequential()
	--[[
	resBlock:add(ShareGradInput(nn.SpatialBatchNormalization(cnn_filters[i]),"preact"))
	layers.add_af(resBlock)
	]]--
	--layers.add_bn(resBlock,cnn_filters[i])
	local s = nn.Sequential()
	layers.add_cnn(s,i,0)
	layers.add_bn(s,cnn_filters[i])
	layers.add_af(s)
	layers.add_cnn(s,i,1)

	return model:add(
		resBlock
		:add(nn.ConcatTable()
		:add(s)
		:add(shortcut()))
		:add(nn.CAddTable(true)))
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
		layers.add_af(model)
		layers.add_bn(model,cnn_filters[i])
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

	models.init(oneByOne)
	models.init(all)
	models.init(model)

	splitter:add(all)
	splitter:add(oneByOne)
	model:add(splitter)
	model:add(nn.JoinTable(1))
	model:add(nn.Sigmoid())

	return model 
end

function models.resNet1()

	--j

	 nFeats = params.nFeats 
	 nLayers = params.nLayers 
	 cnn_filters = torch.range(nFeats,nFeats*nLayers,nFeats)
	 cnn_filters:fill(nFeats)
	 cnn_filter_size = torch.Tensor(nLayers):fill(3)
	 cnn_filter_size[1] = 2 
	 cnn_filter_size[cnn_filter_size:size(1)] = 2
	 cnn_stride = torch.Tensor(nLayers):fill(1)
	 cnn_padding = torch.Tensor(nLayers):fill(1)
	 cnn_padding[cnn_padding:size(1)] = 0
	
	 mp_filter_size = torch.Tensor(nLayers):fill(3)
	 mp_stride = torch.Tensor(nLayers):fill(2)
	 mp_padding = torch.Tensor(nLayers):fill(0)
	
	model = nn.Sequential()
	
	j = 0
	layers.add_cnn(model,1)
	layers.add_bn(model,cnn_filters[1])
	layers.add_af(model)
	layers.add_mp(model,1)
	for i = 2, nLayers -1 do
		j = j + 1
		residualBlock(model,i)
		layers.add_bn(model,cnn_filters[i])
		layers.add_mp(model,i)
	end
	layers.add_cnn(model,cnn_filters:size(1))
	layers.add_bn(model,cnn_filters[{-1}])
	layers.add_af(model)
	layers.add_cnn(model,cnn_filters:size(1))
	layers.add_bn(model,cnn_filters[{-1}])

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

	init(oneByOne)
	init(all)
	init(model)

	splitter:add(all)
	splitter:add(oneByOne)
	model:add(splitter)
	model:add(nn.JoinTable(1))
	model:add(nn.Sigmoid())

	init(model)
	return model
end


function models.main()
	require "cunn"
	params = {} 
	params.windowSize = 256 
	params.nFeats = 16
	params.nWindows = 10 
	params.nLayers = 7 
	model = models.resNet1():cuda()
	print(model)
	input = torch.randn(params.nWindows,3,params.windowSize,params.windowSize):cuda()
	print(input:size())

	o = model:forward(input)
	print(o:size())
end

return models



		
	


	
