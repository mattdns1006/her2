require 'nn'
require 'cutorch'
require "cunn"

local layers = {}

----------------------------------------------------------------------------------------------------------------------
--Define layer functions to add CNN layer or add MP layer to take above paramters
----------------------------------------------------------------------------------------------------------------------
function layers.add_af(model)
	-- return model:add(nn.Tanh())
	return model:add(nn.ReLU())
	-- return model:add(nn.PReLU())
	--return model:add(nn.Sigmoid())
end

function layers.add_bn(model,nFeats)
	return model:add(nn.SpatialBatchNormalization(nFeats))
end

function layers.add_cnn(model,cnn_layer_no,sameSize)
    --First check to see if we are using the first layer;
    if cnn_layer_no > 1 then
	    if sameSize == 1 then
		input_size0 = cnn_filters[cnn_layer_no] -- for back to back convolutions
            else
		input_size0 = cnn_filters[cnn_layer_no-1]
	    end
    else 
	input_size0 = 3 --RGB
    end
    return model:add(nn.SpatialConvolution(input_size0,cnn_filters[cnn_layer_no],
	    cnn_filter_size[cnn_layer_no],cnn_filter_size[cnn_layer_no],
	cnn_stride[cnn_layer_no],cnn_stride[cnn_layer_no],cnn_padding[cnn_layer_no],cnn_padding[cnn_layer_no]))
end


function layers.add_mp(model,mp_layer_no) 
    return model:add(nn.SpatialMaxPooling(mp_filter_size[mp_layer_no],mp_filter_size[mp_layer_no],mp_stride[mp_layer_no],
	mp_stride[mp_layer_no],mp_padding[mp_layer_no],mp_padding[mp_layer_no]))
end

function layers.add_fmp(model,layer_no) 
    return model:add(nn.SpatialFractionalMaxPooling(fmp_filter_size[layer_no],fmp_filter_size[layer_no],
           fmp_output_ratio[layer_no],fmp_output_ratio[layer_no]))
end

function layers.add_fmp_explicit(model,layer_no) 
    return model:add(nn.SpatialFractionalMaxPooling(fmp_filter_size[layer_no],fmp_filter_size[layer_no],
           fmp_output_w[layer_no],fmp_output_h[layer_no]))
end


return layers

