-- Libraries
require "image"
require "nn"
require "gnuplot"
require "cunn"
require "cutorch"


-- Command line options
cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-nThreads",4,"Number of threads to load data.")
cmd:option("-nWindows",10,"Number of windows/ROI.")
cmd:option("-windowSize",200,"Size of ROI.")
cmd:option("-cuda",1,"Use GPU?")
cmd:text()
params = cmd:parse(arg)

dofile("donkeys.lua")
models = require "models"
model = models.fmp0()
print("==> model")
print(model)

count = 1
data = {}
while true do 
donkeys:addjob(function()
			return loadData.loadXY(params.nWindows,params.windowSize)
		end,
		function(Xy)
			data[count] = Xy
			print(#data)
		end
		)
		if #data == 1 then break end
end

imgs = data[1]["data"]
