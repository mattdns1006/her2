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
cmd:option("-nThreads",1,"Number of threads to load data")
cmd:text()
params = cmd:parse(arg)

dofile("donkeys.lua")

for i = 1, 10 do 
donkeys:addjob(function()
		end,
		function()
		end
		)
end
