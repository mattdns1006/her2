
losses = {}
count = count or 1 
local countPrint = 20
require "nn"

function train(inputs,y,coverage)

	if i == nil then
		if model then parameters,gradParameters = model:getParameters() end
		ma = MovingAverage.new(params.ma)
	end
	
	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		gradParameters:zero()

		if params.cuda == 1 then
			inputs = inputs:cuda()
			--y = torch.DoubleTensor{y}:cuda()
			target = torch.zeros(params.nWindows + 1):fill(y):cuda()
			--y = y:cuda()
			

		end

		outputs = model:forward(inputs)
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
		print("==> Output size")
		print(target:size())
	end
	if count % countPrint == 0 then

		lossesT = torch.Tensor(losses)
		print(string.format("Count %d ==> Target = %f, prediciton %f, current loss %f, ma loss %f, coverage %f.",count, target:mean(), outputs:mean(), loss, lossesT[{{-countPrint,-1}}]:mean(),coverage))

		if count > params.ma and params.displayGraph == 1 and count % params.displayGraphFreq ==0 then 
			local MA = ma:forward(lossesT)
			MA:resize(MA:size(1))
			local t = torch.range(1,MA:size(1))
			gnuplot.plot({"Training loss ma of " .. params.ma ,t,MA})
		end
		collectgarbage()
       	end
	if count % params.lrChange == 0 then
		local clr = params.lr
		params.lr = params.lr/params.lrDecay
		print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
	end
	xlua.progress(count,10000)
	count = count + 1
end


