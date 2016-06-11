
local countPrint = 10 

function maMean(tensor)
	return	tensor[{{-params.ma,-1}}]:mean()
end

function train(inputs,target,caseNo)

	if i == nil then
		if model then parameters,gradParameters = model:getParameters() end
		print("Number of parameters ==>")
		print(parameters:size())

		i = 1
	end
	
	local outputs 
	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		gradParameters:zero()
		outputs = model:forward(inputs) -- Only one input for training unlike testing
		local loss = criterion:forward(outputs,target)
		local dLoss_dO = criterion:backward(outputs,target)
		model:backward(inputs,dLoss_dO)

		return	loss, gradParameters 
	end

	_, batchLoss = optimMethod(feval,parameters,optimState)
	local loss = batchLoss[1]

	--[[
	if count % params.displayGraphFreq == 0 then
		print("==> Input size")
		print(inputs)
		print("==> Target")
		print(target)
		print("==> Prediction")
		print(outputs)
	end


	if count % countPrint == 0 then

		targetScore, targetPercScore = target:squeeze()[1], target:squeeze()[2]
		predScore, predPercScore = outputs:squeeze()[1], outputs:squeeze()[2]
		print(string.format("Count %d ==> Targets = {%f, %f}, prediciton {%f, %f}, current loss %f.",count, targetScore, targetPercScore, predScore, predPercScore, loss ))


       	end
	]]--
	if count % params.lrChange == 0 then
		print("==> Saving model " .. modelName .. ".")
		torch.save(modelPath,model)
		local clr = params.lr
		params.lr = params.lr/params.lrDecay
		print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
	end
	xlua.progress(count,params.nIter)
	collectgarbage()
	return loss, outputs

end

function test(inputs,target)

	local outputs = model:forward(inputs)
	local loss = criterion:forward(outputs,target)
	return loss, outputs, target 
end


