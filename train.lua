
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
		outputs:clamp(0.001,0.999)
		local loss = criterion:forward(outputs,target)
		local dLoss_dO = criterion:backward(outputs,target)
		model:backward(inputs,dLoss_dO)

		return	loss, gradParameters 
	end

	_, batchLoss = optimMethod(feval,parameters,optimState)
	local loss = batchLoss[1]

	if count % params.lrChange == 0 then
		print("==> Saving model " .. modelName .. ".")
		torch.save(modelPath,model)
		local clr = params.lr
		params.lr = params.lr/params.lrDecay
		optimState = {
			learningRate = params.lr,
			beta1 = 0.9,
			beta2 = 0.999,
			epsilon = 1e-8
		}
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


