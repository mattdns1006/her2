
losses = {}
count = count or 1 

function train(X,y)

	function feval(x)
		if x~= parameters then parameters:copy(x) end
		gradParameters:zero()

		if params.cuda == 1 then
			X = X:cuda()
			y = torch.LongTensor{y}:cuda()
		end

		outputs = model:forward(X)
		loss = criterion:forward(outputs,y)
		losses[count] = loss
		dLoss_dO = criterion:backward(outputs,y)
		model:backward(X,dLoss_dO)

		return	loss, gradParameters 
	end
	_, batchLoss = optimMethod(feval,parameters,optimState)

	if count % 10 == 0 then
		lossesT = torch.Tensor(losses)
		print("mean loss", lossesT:mean())
		t = torch.range(1,lossesT:size(1))
		--gnuplot.plot({"Training loss",t,lossesT})
		collectgarbage()
	end
	
	--print(count,loss)
	count = count + 1
end


