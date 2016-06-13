-- Libraries
require "image"
require "nn"
require "gnuplot"
require "cunn"
require "xlua"
require "optim"
require "gnuplot"
dofile("movingAverage.lua")

-- Command line options
cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")

cmd:option("-nThreads",10,"Number of threads to load data.")
cmd:option("-nHER2Windows",10,"Number of windows.")
cmd:option("-nHEWindows",4,"Number of windows.")
cmd:option("-windowSize",216,"Size of ROI.")
cmd:option("-level",3,"What level to read images.")
cmd:option("-cuda",1,"Use GPU?")
cmd:option("-run",1,"Run main function.")
cmd:option("-test",0,"Train or test.")
cmd:option("-modelName","resNet1.model","Model name.")
cmd:option("-loadModel",0,"Load model.")
cmd:option("-nTestPreds",20,"Number of different inputs to make predictions on in test.")

cmd:option("-actualTest",0,"Actual test - evaluated test set.")

cmd:option("-display",0,"Display images.")
cmd:option("-displayFreq",80,"Display images.")
cmd:option("-displayGraph",0,"Display graph.")
cmd:option("-displayGraphFreq",200,"Display graph frequency.")
cmd:option("-ma",25,"Moving average.")

cmd:option("-nFeats",24,"Number of features.")
cmd:option("-nLayers",7,"Number of combinations of CNN/BN/AF/MP.")
cmd:option("-checkModel",0,"Runs model with one set of inputs for check.")
cmd:option("-depth",50,"Depth of resnet.")
cmd:option("-shortcut","C","Shortcut.")
cmd:option("-shortcutType","C","Shortcut type.")

cmd:option("-lr",0.0008,"Learning rate.")
cmd:option("-lrDecay",1.1,"Learning rate decay")
cmd:option("-lrChange",2000,"Learning rate change frequency.")
cmd:option("-nIter",30000,"Number of iterations.")
cmd:text()
params = cmd:parse(arg)
print(params)

dofile("donkeys.lua")
dofile("train.lua")
dofile("round.lua")
dofile("counter.lua")
dofile("resultsTable.lua")

optimState = {
	learningRate = params.lr,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-8
}
optimMethod = optim.adam

function display(Xy,outputs,count)
	if params.display == 1 then 
		if imgDisplayHER2 == nil then 
			local initPic = torch.range(1,torch.pow(params.windowSize,2),1):reshape(params.windowSize,params.windowSize)
			imgDisplayHER2 = image.display{image=initPic, zoom=2, offscreen=false}
			imgDisplayHE = image.display{image=initPic, zoom=2, offscreen=false}
		end
		if count % params.displayFreq == 0 then 
			local title = string.format("Case number %d, Targets {%f,%f}, predictions {%f,%f}.", Xy.caseNo, Xy.score, Xy.percScore, 
							round(outputs:squeeze()[1],3), round(outputs:squeeze()[2],3))
			image.display{image = Xy.data[1], win = imgDisplayHER2, legend = title}
			image.display{image = Xy.data[2], win = imgDisplayHE, legend = title}
		end
	end
end

criterion = nn.MSECriterion()
local resModels2 = require "resModels2"
modelName = string.format("%s_%d_%d_%d_%d_%d_%d",params.modelName, params.level, params.nHER2Windows, params.nHEWindows,params.windowSize, params.nFeats, params.nIter)
print(string.format("Model %s, for level %d,  %d features, with %d windows,(%d, %d} window size (sqrt(area))",
		    modelName,params.level, params.nFeats, params.nHER2Windows, params.nHEWindows, params.windowSize))

modelPath = "models/" .. modelName
if params.loadModel == 1 then
	print("==> Loading model "..modelName..".")
	model = torch.load(modelPath)
else
	model = resModels2.resNetSiamese()
end

if params.cuda == 1 then print("==> Placing model on GPU"); model:cuda(); criterion:cuda(); end
--print("==> model"); print(model);

function tensorToMA(tensor,trainOrTest)
	local MA
	if trainOrTest == "train" then 
		MA = ma:forward(tensor)
	else
		MA = testMa:forward(tensor)
	end
	MA:resize(MA:size(1))
	return MA
end

function maMean(table)
	local tensor = torch.Tensor(table)
	return tensor[{{-params.ma,-1}}]:mean()
end

function run() 
	counter = Counter.new()
	testCounter = Counter.new()
	count = count or 1 
	testCount = testCount or 1 
	testCheckCounter = 1
	testResults = ResultsTable.new()
	trainLosses = {}
	testLosses = {}
	trainLossesMA = {}
	testLossesMA = {}
	ma = MovingAverage.new(params.ma)
	testMa = MovingAverage.new(params.ma)
	while true do 
	donkeys:addjob(function()
				return loadData.loadXY(params.nWindows,params.windowSize)
			end,
			function(Xy,tid)
				local y = {}
				local inputs
				local target
				local caseNo
				inputs, target, y.score, y.percScore, caseNo = Xy["data"], Xy["target"], Xy["score"], Xy["percScore"], Xy["caseNo"]

				if params.test == 0 then
					--Training
					if tid ~= 1 then

						local trainLoss, outputs =  train(inputs,target,caseNo)
						trainLosses[#trainLosses + 1] = trainLoss 
						count = count + 1
						display(Xy,outputs,count)
						counter:add(caseNo)
					else
						--Train with test on one thread

						local testLoss, testOutputs, testTarget = test(inputs,target)
						testLosses[#testLosses + 1] = testLoss 
						testCount = testCount + 1 
						testCounter:add(caseNo)
					end

					-- Metrics
					if count > params.ma and testCount > params.ma and count % 40 == 0 then 
						local trainTensor = torch.Tensor(trainLosses)
						local testTensor = torch.Tensor(testLosses)
						local maTrain = trainTensor[{{-params.ma,-1}}]:mean()
						local maTest = testTensor[{{-params.ma,-1}}]:mean()
						trainLossesMA[#trainLossesMA+1] = maTrain
						testLossesMA[#testLossesMA+1] = maTest
						print(string.format("Train/test ma of {%d,%d} = {%f,%f}",params.ma,params.ma,
								maTrain, maTest
								)
								)
						
						-- Graph
						if  params.displayGraph == 1 and  count % params.displayGraphFreq == 0 then 
							local trainLossesMA, testLossesMA = torch.Tensor(trainLossesMA), torch.Tensor(testLossesMA)
							local tTrain, tTest = torch.range(1,trainLossesMA:size(1)), torch.range(1,testLossesMA:size(1))

							gnuplot.plot(
							   {'Train',  tTrain, trainLossesMA,  '-'},
							      {'Test', tTest, testLossesMA, '-'})
							gnuplot.xlabel('time')
							gnuplot.ylabel('loss')
							gnuplot.plotflush()
						end
					end


				else 

					--Testing
						local testLoss, testOutputs, testTarget = test(inputs,target)

						--print(testLoss,caseNo)
						testResults:add(caseNo,testOutputs,testTarget)
						testLosses[#testLosses + 1] = testLoss 
						testCount = testCount + 1 
						testCounter:add(caseNo)
						if testResults:checkCount(testCheckCounter) == true and tableLength(testCounter) == 16 then
							print("Average results after "..testCheckCounter .. " predictions ==>")
							print(testResults:averagePrediction("mean"))
							testCheckCounter = testCheckCounter + 1
						end
						display(Xy,testOutputs,testCount)
				end	

			end

			)
			if params.test == 1 and testCheckCounter == params.nTestPreds then print("Finished testing"); break end
			if count % 500 ==0 then print("Train examples"); print(counter); print("Test examples"); print(testCounter); end
			if count == params.nIter then print("Finished training, saving model."); torch.save(modelPath,model); break; end


	end

end
if params.run == 1 then run() end

