layers = require "layers"

models = {}

local function ConvInit(name)
	for k,v in pairs(model:findModules(name)) do
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

local function BNInit(name)
	for k,v in pairs(model:findModules(name)) do
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
	 nLayers = 8 
	 cnn_filters = torch.range(nFeats,nFeats*nLayers,nFeats)
	 cnn_filters:fill(nFeats)
	 cnn_filter_size = torch.Tensor(nLayers):fill(3)
	 cnn_stride = torch.Tensor(nLayers):fill(1)
	 cnn_padding = torch.Tensor(nLayers):fill(1)
	
	 mp_filter_size = torch.Tensor(nLayers):fill(3)
	 mp_stride = torch.Tensor(nLayers):fill(2)
	 mp_padding = torch.Tensor(nLayers):fill(0)
	
	model = nn.Sequential()
	
	j = 0
	model:add(nn.SpatialBatchNormalization(3))
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
	local nOutputsDense = outputSize[2]*outputSize[3]*outputSize[4] 
	local feats = 10

	model:add(nn.View(nOutputsDense*params.nWindows)) -- Full combine
	model:add(nn.Linear(nOutputsDense*params.nWindows,feats))
	layers.add_af(model)
	--[[
	model:add(nn.Linear(feats,feats))
	layers.add_af(model)
	model:add(nn.Linear(feats,feats))
	layers.add_af(model)
	]]--
	model:add(nn.Linear(feats,1))
	model:add(nn.Sigmoid())

	ConvInit('cudnn.SpatialConvolution')
   	ConvInit('nn.SpatialConvolution')
      	BNInit('fbnn.SpatialBatchNormalization')
	BNInit('cudnn.SpatialBatchNormalization')
	BNInit('nn.SpatialBatchNormalization')
	for k,v in pairs(model:findModules('nn.Linear')) do
	      v.bias:zero()
	end
	return model
end



function models.main()
	params = {} 
	params.windowSize = 512 
	params.nFeats = 16
	params.nWindows = 4 
	model = models.model1()
	print(model)
	input = torch.randn(params.nWindows,3,params.windowSize,params.windowSize)

	print(model:forward(input):size())
end

return models



		
	


	
