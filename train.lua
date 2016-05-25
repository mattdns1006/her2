losses = {}
count = count or 1 
local countPrint = 20

function train(inputs,target,caseNo,coverage)

	if i == nil then
		if model then parameters,gradParameters = model:getParameters() end
		print("Number of parameters ==>")
		print(parameters:size())
		ma = MovingAverage.new(params.ma)
		i = 1
	end
	
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
		print("==> Some parameters")
		print(model:parameters()[1][1])
		print("==> Input size")
		print(inputs:size())
		print("==> Target")
		print(target)
		print("==> Prediction")
		print(outputs)
	end
	if count % countPrint == 0 then

		local lossesT = torch.Tensor(losses)
		local targets = target:mean(1):squeeze()
		local predictions = outputs:mean(1):squeeze()
		print(string.format("Count %d ==> Targets = {%f, %f}, prediciton {%f, %f}, current loss %f, ma loss %f, coverage %f.",count, targets[1], targets[2], predictions[1], predictions[2], loss, lossesT[{{-countPrint,-1}}]:mean(),coverage))

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
		torch.save("models/"..modelName,model)
		local clr = params.lr
		params.lr = params.lr/params.lrDecay
		print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
	end
	xlua.progress(count,params.nIter)
	count = count + 1
end

function test(inputs,target,caseNo)

	local outputs = model:forward(inputs)
	local loss = criterion:forward(outputs,target)

	local testOutput = {}
	testOutput[1] = caseNo
	testOutput[2] = loss 
	testOutput[3] = outputs
	testOutput[4] = target 
	
	return testOutput
end


