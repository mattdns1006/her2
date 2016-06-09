losses = {}
count = count or 1 
local countPrint = 10 

function train(inputs,target,caseNo)

	if i == nil then
		if model then parameters,gradParameters = model:getParameters() end
		print("Number of parameters ==>")
		print(parameters:size())
		ma = MovingAverage.new(params.ma)
		i = 1
	end
	
	local loss
	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		gradParameters:zero()
		outputs = model:forward(inputs) -- Only one input for training unlike testing
		loss = criterion:forward(outputs,target)
		losses[count] = loss
		dLoss_dO = criterion:backward(outputs,target)
		model:backward(inputs,dLoss_dO)

		return	loss, gradParameters 
	end

	_, batchLoss = optimMethod(feval,parameters,optimState)

	if count % params.displayGraphFreq == 0 then
		print("==> Input size")
		print(inputs)
		print("==> Target")
		print(target)
		print("==> Prediction")
		print(outputs)
	end
	if count % countPrint == 0 then

		lossesT = torch.Tensor(losses)
		targetScore, targetPercScore = target:squeeze()[1], target:squeeze()[2]
		predScore, predPercScore = outputs:squeeze()[1], outputs:squeeze()[2]
		print(string.format("Count %d ==> Targets = {%f, %f}, prediciton {%f, %f}, current loss %f, ma loss %f.",count, targetScore, targetPercScore, predScore, predPercScore, loss, lossesT[{{-countPrint,-1}}]:mean()))

		if count > params.ma and params.displayGraph == 1 and count % params.displayGraphFreq ==0 then 
			local MA = ma:forward(lossesT)
			MA:resize(MA:size(1))
			local t = torch.range(1,MA:size(1))
			local title = string.format("Model %s has ma mean (%d) training loss of % f",modelName, params.ma, MA:mean())
			gnuplot.plot({title,t,MA})
		end
		collectgarbage()
       	end
	if count % params.lrChange == 0 then
		print("==> Saving model " .. modelName .. ".")
		torch.save(modelPath,model)
		local clr = params.lr
		params.lr = params.lr/params.lrDecay
		print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
	end
	xlua.progress(count,params.nIter)

end

function test(inputs,target,caseNo)

	if testCount == nil then
		testCount = 0
	end

	local outputs = model:forward(inputs)
	local loss = criterion:forward(outputs,target)

	local testOutput = {}
	testOutput[1] = caseNo
	testOutput[2] = loss 
	testOutput[3] = outputs
	testOutput[4] = target 
	xlua.progress(testCount,16*params.nTestPreds)
	testCount = testCount + 1
	
	return testOutput
end


