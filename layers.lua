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

------------------------------------------------------------------------------------
			---- Residual Blocks ----
------------------------------------------------------------------------------------
--
--
Convolution = nn.SpatialConvolution
Avg = nn.SpatialAveragePooling
ReLU = nn.ReLU
Max = nn.SpatialMaxPooling
SBatchNorm = nn.SpatialBatchNormalization

   -- The shortcut layer is either identity or 1x1 convolution
function layers.shortcut(nInputPlane, nOutputPlane, stride)
   local useConv = shortcutType == 'C' or
      (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
   if useConv then
      -- 1x1 convolution
      return nn.Sequential()
         :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
         :add(SBatchNorm(nOutputPlane))
   elseif nInputPlane ~= nOutputPlane then
      -- Strided, zero-padded identity shortcut
      return nn.Sequential()
         :add(nn.SpatialAveragePooling(1, 1, stride, stride))
         :add(nn.Concat(2)
            :add(nn.Identity())
            :add(nn.MulConstant(0)))
   else
      return nn.Identity()
   end
end

-- The basic residual layer block for 18 and 34 layer network, and the
-- CIFAR networks
function layers.basicblock(n, stride)
   local nInputPlane = iChannels
   iChannels = n

   local s = nn.Sequential()
   s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
   s:add(SBatchNorm(n))
   s:add(ReLU(true))
   s:add(Convolution(n,n,3,3,1,1,1,1))
   s:add(SBatchNorm(n))

   return nn.Sequential()
      :add(nn.ConcatTable()
         :add(s)
         :add(layers.shortcut(nInputPlane, n, stride)))
      :add(nn.CAddTable(true))
      :add(ReLU(true))
end

-- The bottleneck residual layer for 50, 101, and 152 layer networks
function layers.bottleneck(n, stride)
   local nInputPlane = iChannels
   iChannels = n * 4

   local s = nn.Sequential()
   s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
   s:add(SBatchNorm(n))
   s:add(ReLU(true))
   s:add(Convolution(n,n,3,3,stride,stride,1,1))
   s:add(SBatchNorm(n))
   s:add(ReLU(true))
   s:add(Convolution(n,n*4,1,1,1,1,0,0))
   s:add(SBatchNorm(n * 4))

   return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(layers.shortcut(nInputPlane, n * 4, stride)))
         :add(nn.CAddTable(true))
         :add(ReLU(true))
end

-- Creates count residual blocks with specified number of features
function layers.layer(block, features, count, stride, iChannels)
	local s = nn.Sequential()
	for i=1,count do
		 s:add(block(features, i == 1 and stride or 1,iChannels))
	end
	return s
end

------------------------------------------------------------------------------------
			---- Initialization Layers ----
------------------------------------------------------------------------------------

function layers.ConvInit(name,block)
	for k,v in pairs(block:findModules(name)) do
		local n = v.kW*v.kH*v.nOutputPlane
		v.weight:normal(0,math.sqrt(2/n))
		v.bias:zero()
	end
end

function layers.BNInit(name,block)
	for k,v in pairs(block:findModules(name)) do
		v.weight:fill(1)
		v.bias:zero()
	end
end

function layers.linearInit(block)
	for k,v in pairs(block:findModules('nn.Linear')) do
	      v.bias:zero()
	end
end

function layers.init(block)
	layers.ConvInit('cudnn.SpatialConvolution',block)
	layers.ConvInit('nn.SpatialConvolution',block)
	layers.BNInit('fbnn.SpatialBatchNormalization',block)
	layers.BNInit('cudnn.SpatialBatchNormalization',block)
	layers.BNInit('nn.SpatialBatchNormalization',block)
	layers.linearInit(block)
end

return layers

