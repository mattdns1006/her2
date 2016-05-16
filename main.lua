-- Libraries
require "image"
require "nn"
require "gnuplot"
require "cunn"
--require "cutorch"
require "xlua"
require "optim"
require "gnuplot"


-- Command line options
cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-nThreads",10,"Number of threads to load data.")
cmd:option("-nWindows",1,"Number of windows/ROI.")
cmd:option("-windowSize",2048,"Size of ROI.")
cmd:option("-cuda",1,"Use GPU?")
cmd:option("-run",1,"Run main function.")
cmd:option("-display",0,"Display images.")
cmd:option("-displayGraph",0,"Display graph.")
cmd:option("-displayFreq",50,"Display images.")
cmd:option("-lr",0.0001,"Learning rate.")
cmd:text()
params = cmd:parse(arg)

dofile("donkeys.lua")
dofile("train.lua")

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
			imgDisplay = image.display{image=initPic, zoom=1, offscreen=false}
		end
		if count % params.displayFreq == 0 then 
			image.display{image = X, win = imgDisplay, legend = "Truth ".. y.." prediction ".. outputs[1]}
		end
	end
end

criterion = nn.MSECriterion()
models = require "models"
model = models.model1()
parameters, gradParameters = model:getParameters()

if params.cuda == 1 then
	print("==> Placing model on GPU")
	model:cuda()
	criterion:cuda()
end

print("==> model")
print(model)


function run() 

	while true do 
	donkeys:addjob(function()
				return loadData.loadXY(params.nWindows,params.windowSize)
			end,
			function(Xy)
				X,y = Xy["data"], Xy["score"]
				train(X,y)
				display(X,y,outputs)
			end
			)
			--if count == 50 then break end
	end

end
if params.run == 1 then run() end

