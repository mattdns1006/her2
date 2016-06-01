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
cmd:option("-nWindows",100,"Number of windows at level 4.")
cmd:option("-windowSize",216,"Size of ROI.")
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
cmd:option("-lrChange",2000,"Learning rate change frequency.")
cmd:option("-nIter",50000,"Number of iterations.")
cmd:text()
params = cmd:parse(arg)


params.nLevelAdjust = 216/params.windowSize - 1
local level4winSize = params.windowSize 
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

modelPath = "models/" .. modelName
if params.loadModel == 1 then
	print("==> Loading model "..modelName..".")
	model = torch.load(modelPath)
else
	model = resModels.resNet()
end

if params.cuda == 1 then print("==> Placing model on GPU"); model:cuda(); criterion:cuda(); end
print("==> model"); print(model);

function checkModel()
	local loadData = require "loadData"
	loadData.init(1,1,params.level)
	Xy = loadData.loadXY(params.nWindows,params.windowSize)
	y = {}
	inputs, target, y.score, y.percScore, coverage, caseNo = Xy["data"], Xy["target"], Xy["score"], Xy["percScore"], Xy["coverage"], Xy["caseNo"]
	outputs = model:forward(inputs)
	print("Input/Output sizes")
	print(inputs:size())
	print(outputs:size())
end
if params.checkModel == 1 then checkModel(); params.run = 0; end

function run() 
	counter = Counter.new()
	if params.test == 1 then
		testResults = ResultsTable.new()
		nTestPreds = 1
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
					caseNo, loss, outputs, target = table.unpack(testOutput)
					testResults:add(tostring(caseNo),{loss,outputs[1]:reshape(1,2),target}) -- Use all or ony by one predictions 
					--testResults:add(tostring(caseNo),{loss,outputs:mean(1),target})

					if tableLength(testResults) == 16 and testResults:checkCount(nTestPreds) == true then
						print(string.format("Average test result using first %d predictions",nTestPreds))
						print(testResults:averageLoss(nTestPreds)["average"])
						nTestPreds = nTestPreds + 1

						if testResults:checkCount(params.nTestPreds) == true then
							print("Finished testing")
							finishedTesting = true
						end

					end
		
				end
				count = count + 1
				display(Xy)
				counter:add(caseNo)

			end
			)
			if params.test == 1 and finishedTesting == true then break; end
			if count % params.displayGraphFreq ==0 then print(counter) end
			if count == params.nIter and params.test ==0 then print("Finished training, saving model."); torch.save(modelPath,model); break; end

	end

end
if params.run == 1 then run() end

