require "image"
require "gnuplot"
require "nn"
require "cunn"
--require "cutorch"
require "xlua"
require "optim"
require "gnuplot"
dofile("../movingAverage.lua")
dofile("loadData.lua")
dofile("train.lua")
dofile("binaryConfusionMatrix.lua")

cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-windowSize",128,"WindowSize.")
cmd:option("-modelName","filter.model","Name of model.")
cmd:option("-modelSave",2000,"How often to save.")
cmd:option("-loadModel",0,"Load model.")

cmd:option("-lr",0.00003,"Learning rate.")
cmd:option("-lrDecay",1.01,"Learning rate change factor.")
cmd:option("-lrChange",2000,"How often to change lr.")

cmd:option("-display",1,"Display images.")
cmd:option("-displayFreq",100,"Display images frequency.")
cmd:option("-displayGraph",0,"Display graph of loss.")
cmd:option("-displayGraphFreq",500,"Display graph of loss.")
cmd:option("-nIter",100000,"Number of iterations.")
cmd:option("-ma",200,"Moving average.")
cmd:option("-run",1,"Run.")

cmd:option("-cmThresh",0.5,"Confusion matrix threshold.")
cmd:option("-rocInterval",0.1,"ROC intervals.")

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

function display(x,y,output)
	if params.display == 1 then 
		if imgDisplay == nil then 
			local initPic = torch.range(1,torch.pow(params.windowSize,2),1):reshape(params.windowSize,params.windowSize)
			imgDisplay0 = image.display{image=initPic, zoom=2, offscreen=false}
			imgDisplay1 = image.display{image=initPic, zoom=2, offscreen=false}
			imgDisplay = 1 
		end
		if i % params.displayFreq == 0 and y[1] == 0 then 
			local title = string.format("Target/prediction %d, %f .", y[1], output[1])
			image.display{image = x, win = imgDisplay0, legend = title}
		end
		if i % params.displayFreq == 0 and y[1] == 1 then 
			local title = string.format("Target/prediction %d, %f .", y[1], output[1])
			image.display{image = x, win = imgDisplay1, legend = title}
		end

	end
end



local function buildModel()
	local layers = dofile("../layers.lua")

	local Convolution = nn.SpatialConvolution
	local Pool = nn.SpatialMaxPooling
	local SBN = nn.SpatialBatchNormalization
	local af = nn.ReLU
	local Linear = nn.Linear
	local Dropout = nn.Dropout
	local function block(model)
		local j = j or 0.01
		nInputs = nOutputs or 3
		nOutputs = nOutputs or 64 
		model
		--:add(Dropout(j*2))
		:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		:add(SBN(nOutputs))
		:add(af())
		:add(Pool(3,3,2,2,0,0))
	end
		
	local model = nn.Sequential()
	for i = 1, 5 do
		block(model)
	end
	--model:add(Convolution(nInputs,nOutputs,3,3,2,2,1,1))
	local reshapeOutputSize = model:forward(torch.randn(1,3,params.windowSize,params.windowSize)):size()
	local nOutputsDense = reshapeOutputSize[2]*reshapeOutputSize[3]*reshapeOutputSize[4]
	print(reshapeOutputSize)
	model:add(nn.View(nOutputsDense))
	local nDenseConnections = 20
	model:add(Linear(nOutputsDense,nDenseConnections))
	model:add(af())
	model:add(Linear(nDenseConnections,nDenseConnections))
	model:add(af())
	model:add(Linear(nDenseConnections,1))
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
cm = BinaryConfusionMatrix.new(params.cmThresh,params.rocInterval)

function onePass()
	if i == nil then i = 1 end
	x,y = loadData.loadObs()
	x,y = x:cuda(), torch.Tensor{y}:cuda()
	train(x,y)
	if params.display == 1 then
		display(x,y,output)
	end
end


function run()
	if i == nil then i = 1 end
	while i < params.nIter do
		onePass()
	end
end
if params.run == 1 then run() end

function test()
	for j=1,81 do 
		img = image.loadJPG("../data/roi_61/3/"..j..".jpg")
		imgScale = image.scale(img,128,128)
		pred = model:forward(imgScale:resize(1,3,128,128):cuda())
		print(j,pred[1])
	end
end
	


















