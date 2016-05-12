layers = require "layers"

models = {}

function models.fmp0()

	 nFeats = 100 
	 cnn_filters = torch.Tensor{nFeats,nFeats*2,nFeats*3,nFeats*4,1}
	 cnn_filter_size = {2,2,2,2,2,2}
	 cnn_stride = {1,1,1,1,1,1}
	 cnn_padding = {0,0,0,0,0,0}
	
	 fmp_filter_size = {2,2,2,2,2,2,2}

	 ratio = 2/3
	 fmp_output_ratio= {ratio,ratio,ratio,ratio,ratio,ratio}
	 fmp_output_w = {20,12,7,4,2}
	 fmp_output_h = {20,12,7,4,2}
	
	model = nn.Sequential()
	
	j = 0
	for i = 1, 4 do
		j = j + 1
		layers.add_cnn(model,i)
		layers.add_af(model)
		layers.add_fmp_explicit(model,i)
		model:add(nn.SpatialBatchNormalization(cnn_filters[i]))
	end

	layers.add_cnn(model,cnn_filters:size(1))
	model:add(nn.SpatialBatchNormalization(cnn_filters[{-1}]))

	local nOutputsDense = cnn_filters[{-1}]*3*3
	model:add(nn.View(-1))
	model:add(nn.Linear(90,1))

	return model
end

return models


