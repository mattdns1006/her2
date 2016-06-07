require "image"
require "gnuplot"
require "nn"
require "cunn"
--require "cutorch"
require "xlua"
require "optim"
require "gnuplot"
dofile("../movingAverage.lua")
loadData = require("loadData")
dofile("train.lua")

cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-modelName","deconv.model","Name of model.")
cmd:option("-modelSave",2000,"How often to save.")
cmd:option("-loadModel",0,"Load model.")

cmd:option("-lr",0.0005,"Learning rate.")
cmd:option("-lrDecay",1.02,"Learning rate change factor.")
cmd:option("-lrChange",1000,"How often to change lr.")

cmd:option("-display",1,"Display images.")
cmd:option("-displayFreq",100,"Display images frequency.")
cmd:option("-displayGraph",0,"Display graph of loss.")
cmd:option("-displayGraphFreq",500,"Display graph of loss.")
cmd:option("-nIter",100000,"Number of iterations.")
cmd:option("-ma",200,"Moving average.")
cmd:option("-run",1,"Run.")


--[[
cmd:option("-",,".")
cmd:option("-",,".")
]]--

cmd:text()
params = cmd:parse(arg)


optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam

function display(x,y,output,train)
	if params.display == 1 then 
		if imgDisplay == nil then 
			local zoom = 4
			local initPic = torch.range(1,torch.pow(100,2),1):reshape(100,100)
			imgDisplay0 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay1 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay2 = image.display{image=initPic, zoom=zoom, offscreen=false}

			imgDisplay3 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay4 = image.display{image=initPic, zoom=zoom, offscreen=false}
			imgDisplay5 = image.display{image=initPic, zoom=zoom, offscreen=false}

			imgDisplay = 1 
		end
		local title
		if train == 1 then
			title = "Train"
			image.display{image = x, win = imgDisplay0, legend = title}
			image.display{image = y, win = imgDisplay1, legend = title}
			image.display{image = output, win = imgDisplay2, legend = title}
		else	
			title = "Test"
			image.display{image = x, win = imgDisplay3, legend = title}
			image.display{image = y, win = imgDisplay4, legend = title}
			image.display{image = output, win = imgDisplay5, legend = title}
		end
	end
end



function buildModel()
	local layers = dofile("../layers.lua")

	local Convolution = nn.SpatialConvolution
	local Pool = nn.SpatialMaxPooling
	local fmp = nn.SpatialFractionalMaxPooling
	local UpSample = nn.SpatialUpSamplingNearest
	local SBN = nn.SpatialBatchNormalization
	local af = nn.ReLU
	local Linear = nn.Linear
	local Dropout = nn.Dropout
	local function same(model)
		nInputs = nOutputs or 3
		nOutputs = nOutputs or 3 
		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		:add(SBN(nOutputs))
		:add(af())
	end
	local function down(model)
		nInputs = nOutputs or 3
		nOutputs = nOutputs or 12 
		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		:add(SBN(nOutputs))
		:add(af())
		:add(Pool(3,3,2,2,1,1))
	end
	local function up(model)
		nInputs = nOutputs or 3
		nOutputs = nOutputs or 12 
		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		:add(SBN(nOutputs))
		:add(af())
		:add(UpSample(2))
	end
		
	local model = nn.Sequential()
	--local testInput = torch.rand(1,3,384,768)
	for i = 1, 6 do down(model); 
	end; for i = 1, 6 do up(model);
	end
	model:add(Convolution(nInputs,1,3,3,1,1,1,1))
	model:add(nn.Sigmoid())
	layers.init(model)

	return model
end

print("Model name ==>")
print(params.modelName)
if params.loadModel == 1 then
	print("==> Loading model")
	model = torch.load(params.modelName):cuda()
else 	
	model = buildModel():cuda()
end
criterion = nn.MSECriterion():cuda()

function onePass()
	if i == nil then i = 1 end
	local x,y = loadData.loadObs("train")
	train(x,y)
	if i % 20 == 0 then
		display(x,y,output,1)
		local x,y = loadData.loadObs("test")
		local output = model:forward(x)
		display(x,y,output,0)
	end
end


function run()
	if i == nil then i = 1 end
	while i < params.nIter do
		onePass()
	end
end
if params.run == 1 then run() end

	


















