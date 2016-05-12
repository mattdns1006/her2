-- Libraries
require "image"
require "nn"
require "gnuplot"
require "cunn"
require "cutorch"
require "xlua"
require "optim"
require "gnuplot"


-- Command line options
cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-nThreads",8,"Number of threads to load data.")
cmd:option("-nWindows",10,"Number of windows/ROI.")
cmd:option("-windowSize",200,"Size of ROI.")
cmd:option("-cuda",1,"Use GPU?")
cmd:option("-run",0,"Run main function.")
cmd:text()
params = cmd:parse(arg)

dofile("donkeys.lua")
dofile("train.lua")

optimState = {
	learningRate = 0.001,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam

criterion = nn.MSECriterion()
models = require "models"
model = models.fmp0()
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
				X,y = Xy["data"], Xy["percScore"]
				train(X,y)
			end
			)
			--if count == 50 then break end
	end

end
if params.run == 1 then run() end

