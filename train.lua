
losses = {}
count = count or 1 
local countPrint = 20

function train(X,y,coverage)

	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		model:zeroGradParameters()

		if params.cuda == 1 then
			X = X:cuda()
			--y = torch.DoubleTensor{y}:cuda()
			target = torch.zeros(params.nWindows + 1):fill(y):cuda()
			--y = y:cuda()
			
		end

		outputs = model:forward(X)
		loss = criterion:forward(outputs,target)
		losses[count] = loss
		dLoss_dO = criterion:backward(outputs,target)
		model:backward(X,dLoss_dO)

		return	loss, gradParameters 
	end
	_, batchLoss = optimMethod(feval,parameters,optimState)


	if count % countPrint == 0 then
		print("==> Input size")
		print(X:size())
		print("==> Output size")
		print(target:size())
		lossesT = torch.Tensor(losses)
		print(string.format("Count %d ==> Target = %f, prediciton %f, current loss %f, ma loss %f, coverage %f.",count, target:mean(), outputs:mean(), loss, lossesT[{{-countPrint,-1}}]:mean(),coverage))
		if params.displayGraph == 1 and count % params.displayGraphFreq ==0 then 
			t = torch.range(1,lossesT:size(1))
			gnuplot.plot({"Training loss",t,lossesT})
		end
		collectgarbage()
	end
	xlua.progress(count,10000)
	count = count + 1
end


