function train(inputs,target)

	if i == 1 then
		if model then parameters,gradParameters = model:getParameters() end
		print("Number of parameters ==>")
		print(parameters:size())
		ma = MovingAverage.new(params.ma)
		losses = {}
	end
	
	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		gradParameters:zero()
		output = model:forward(inputs) -- Only one input for training unlike testing
		target = image.scale(target:double(),output:size(4),output:size(3)):cuda()
		loss = criterion:forward(output,target)
		losses[i] = loss
		dLoss_dO = criterion:backward(output,target)
		model:backward(inputs,dLoss_dO)

		return	loss, gradParameters 
	end

	_, batchLoss = optimMethod(feval,parameters,optimState)

	if i % params.ma == 0 then

		local lossesT = torch.Tensor(losses)
		local MA = ma:forward(lossesT)
		print(MA[{-1}])
		if i > params.ma and params.displayGraph == 1 and i % params.displayGraphFreq ==0 then 
			MA:resize(MA:size(1))
			local t = torch.range(1,MA:size(1))
			local title = string.format("Model %s has ma mean (%d) training loss of % f",modelName, params.ma, MA:mean())
			gnuplot.plot({title,t,MA})
		end
		collectgarbage()
       	end
	if i % params.lrChange == 0 then
		local clr = params.lr
		params.lr = params.lr/params.lrDecay
		print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
	end
	if i % params.modelSave == 0 then
		print("==> Saving model " .. params.modelName .. ".")
		torch.save(params.modelName,model)
	end
	xlua.progress(i,params.nIter)
	i = i + 1

end

