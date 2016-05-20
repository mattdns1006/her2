-- Libraries
require "image"
require "nn"
require "gnuplot"
require "cunn"
--require "cutorch"
require "xlua"
require "optim"
require "gnuplot"

dofile("movingAverage.lua")

-- Command line options
cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-nThreads",10,"Number of threads to load data.")
cmd:option("-nWindows",20,"Number of windows/ROI.")
cmd:option("-windowSize",244,"Size of ROI.")
cmd:option("-level",4,"What level to read images.")
cmd:option("-cuda",1,"Use GPU?")
cmd:option("-run",1,"Run main function.")
cmd:option("-display",0,"Display images.")
cmd:option("-displayFreq",80,"Display images.")
cmd:option("-displayGraph",0,"Display graph.")
cmd:option("-displayGraphFreq",200,"Display graph frequency.")
cmd:option("-ma",20,"Moving average.")
cmd:option("-lr",0.003,"Learning rate.")
cmd:option("-nFeats",16,"Number of features.")
cmd:option("-nLayers",8,"Number of combinations of CNN/BN/AF/MP.")
cmd:text()
params = cmd:parse(arg)

dofile("donkeys.lua")
dofile("train.lua")
dofile("counter.lua")

optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam


function display(X,y,outputs)
	if params.display == 1 then 
		if imgDisplay == nil then 
			local initPic = torch.range(1,torch.pow(params.windowSize,2),1):reshape(params.windowSize,params.windowSize)
			imgDisplay = image.display{image=initPic, zoom=2, offscreen=false}
		end
		if count % params.displayFreq == 0 then 
			image.display{image = X, win = imgDisplay, legend = "Truth ".. y .." prediction ".. outputs[1]}
		end
	end
end

criterion = nn.MSECriterion()
models = require "models"
model = models.model1()

if params.cuda == 1 then
	print("==> Placing model on GPU")
	model:cuda()
	criterion:cuda()
end

print("==> model")
print(model)


function run() 
	counter = Counter.new()
	while true do 
	donkeys:addjob(function()
				return loadData.loadXY(params.nWindows,params.windowSize)
			end,
			function(Xy)
				X,y,coverage = Xy["data"], Xy["score"], Xy["coverage"]
				train(X,y,coverage)
				display(X,y,outputs)
				counter:add(y)
			end
			)
			if count % params.displayGraphFreq ==0 then print(counter) end

	end

end
if params.run == 1 then run() end

