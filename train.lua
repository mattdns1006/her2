
losses = {}
count = count or 1 

function train(X,y)

	function feval(x)
		if x~= parameters then parameters:copy(x) end
		gradParameters:zero()

		if params.cuda == 1 then
			X = X:cuda()
			y = torch.DoubleTensor{y}:cuda()
			--y = y:cuda()
			
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
		print(string.format("Count %d ==> Target = %f, prediciton %f, current loss %f, ma loss %f.",count, y[1], outputs[1], loss, lossesT[{{-10,-1}}]:mean()))
		if params.displayGraph == 1 then 
			t = torch.range(1,lossesT:size(1))
			gnuplot.plot({"Training loss",t,lossesT})
		end
		collectgarbage()
	end
	xlua.progress(count,10000)
	count = count + 1
end


