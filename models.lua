layers = require "layers"

models = {}

function models.model1()

	 nFeats = 8 
	 nLayers = 9 
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
		layers.add_af(model)
		layers.add_mp(model,i)
		model:add(nn.SpatialBatchNormalization(cnn_filters[i]))
	end

	layers.add_cnn(model,cnn_filters:size(1))
	model:add(nn.SpatialBatchNormalization(cnn_filters[{-1}]))
	layers.add_af(model)

	outputSize = model:forward(torch.randn(params.nWindows,3,params.windowSize,params.windowSize)):size()

	nOutputsDense = outputSize[2]*outputSize[3]*outputSize[4] 

	model:add(nn.View(nOutputsDense*params.nWindows))
	model:add(nn.Linear(nOutputsDense*params.nWindows,1))
	model:add(nn.Sigmoid())

	return model
end

function models.main()
	params = {} 
	params.windowSize = 2048 
	params.nWindows = 4 
	model = models.model1()
	print(model)
	input = torch.randn(params.nWindows,3,params.windowSize,params.windowSize)

	print(model:forward(input):size())
end

return models



		
	


	
