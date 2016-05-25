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
cmd:option("-nWindows",32,"Number of windows at level 4.")
--cmd:option("-windowSize",256,"Size of ROI.")
cmd:option("-level",3,"What level to read images.")
cmd:option("-cuda",1,"Use GPU?")
cmd:option("-run",1,"Run main function.")
cmd:option("-test",0,"Train or test.")
cmd:option("-modelName","resNet1.model","Model name.")
cmd:option("-loadModel",0,"Load model.")
cmd:option("-nTestPreds",20,"Number of different inputs to make predictions on in test.")

cmd:option("-display",0,"Display images.")
cmd:option("-displayFreq",80,"Display images.")
cmd:option("-displayGraph",0,"Display graph.")
cmd:option("-displayGraphFreq",200,"Display graph frequency.")
cmd:option("-ma",80,"Moving average.")

cmd:option("-nFeats",24,"Number of features.")
cmd:option("-nLayers",7,"Number of combinations of CNN/BN/AF/MP.")
cmd:option("-checkModel",0,"Runs model with one set of inputs for check.")
cmd:option("-depth",50,"Depth of resnet.")
cmd:option("-shortcut","C","Shortcut.")
cmd:option("-dataset","her2","Dataset.")
cmd:option("-shortcutType","C","Shortcut type.")

cmd:option("-lr",0.0008,"Learning rate.")
cmd:option("-lrDecay",1.1,"Learning rate decay")
cmd:option("-lrChange",400,"Learning rate change frequency.")
cmd:option("-nIter",20000,"Number of iterations.")
cmd:text()
params = cmd:parse(arg)

local level4winSize = 216 
local level4nWindows= params.nWindows 
local levelParams = {

	["0"] = {level4winSize*16,level4nWindows/16},
	["1"] = {level4winSize*8,level4nWindows/8},
	["2"] = {level4winSize*4,level4nWindows/4},
	["3"] = {level4winSize*2,level4nWindows/2},
	["4"] = {level4winSize*1,level4nWindows/1},

}
params.windowSize, params.nWindows, params.nDownsample = table.unpack(levelParams[tostring(params.level)])


dofile("donkeys.lua")
dofile("train.lua")
dofile("counter.lua")
dofile("resultsTable.lua")

optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam


function display(Xy)
	if params.display == 1 then 
		if imgDisplay == nil then 
			local initPic = torch.range(1,torch.pow(params.windowSize,2),1):reshape(params.windowSize,params.windowSize)
			imgDisplay = image.display{image=initPic, zoom=2, offscreen=false}
		end
		if count % params.displayFreq == 0 then 
			local title = string.format("Case number %d, Targets {%f,%f}, predictions {%f,%f}.", Xy.caseNo, Xy.score, Xy.percScore, 
							outputs[{{},{1}}]:mean(), outputs[{{},{2}}]:mean())
			image.display{image = Xy.data, win = imgDisplay, legend = title}
		end
	end
end

criterion = nn.MSECriterion()
resModels = require "resModels"
modelName = string.format("%s_%d_%d_%d_%d_%d",params.modelName, params.level, params.nWindows, params.windowSize, params.nFeats, params.nIter)
print(string.format("Model %s, for level %d,  %d features, with %d windows,%d window size (sqrt(area))",
		    modelName,params.level, params.nFeats, params.nWindows, params.windowSize))

if params.loadModel == 1 then
	print("==> Loading model "..modelName..".")
	model = torch.load("models/"..modelName)
else
	model = resModels.resNet()
end

if params.cuda == 1 then print("==> Placing model on GPU"); model:cuda(); criterion:cuda(); end
print("==> model"); print(model);

function checkModel()
	input = torch.randn(params.nWindows,3,params.windowSize,params.windowSize):cuda()
	print(input:size())
	o = model:forward(input)
	print(o:size())
end
if params.checkModel == 1 then checkModel(); params.run = 0; end

function run() 
	counter = Counter.new()
	if params.test == 1 then
		testResults = ResultsTable.new()
	end
	while true do 
	donkeys:addjob(function()
				return loadData.loadXY(params.nWindows,params.windowSize)
			end,
			function(Xy)
				y = {}
				inputs, target, y.score, y.percScore, coverage, caseNo = Xy["data"], Xy["target"], Xy["score"], Xy["percScore"], Xy["coverage"], Xy["caseNo"]
				if params.test == 0 then
					train(inputs,target,caseNo,coverage)
				else
					local testOutput = test(inputs,target,caseNo)
					local caseNo, loss, outputs, target = table.unpack(testOutput)
					testResults:add(tostring(caseNo),{loss,outputs,target})
					if testResults:checkCount(5) == true then
						print("Finished testing")
						finishedTesting = true
					end
		
				end
				display(Xy)
				counter:add(caseNo)

			end
			)
			if params.test == 1 and finishedTesting == true then break; end
			if count % params.displayGraphFreq ==0 then print(counter) end
			if count == params.nIter and params.test ==0 then print("Finished training, saving model."); torch.save(modelName,model); break; end

	end

end
if params.run == 1 then run() end

