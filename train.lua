losses = {}
count = count or 1 
local countPrint = 20

function train(inputs,target,caseNo,coverage)

	if i == nil then
		if model then parameters,gradParameters = model:getParameters() end
		ma = MovingAverage.new(params.ma)
	end
	
	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		gradParameters:zero()
		outputs = model:forward(inputs[1])
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
			gnuplot.plot({"Training loss ma of " .. params.ma.. "current mean of " .. MA:mean()..".",t,MA})
		end
		collectgarbage()
       	end
	if count % params.lrChange == 0 then
		print("==> Saving model " .. params.modelName .. ".")
		torch.save(params.modelName,model)
		local clr = params.lr
		params.lr = params.lr/params.lrDecay
		print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
	end
	xlua.progress(count,10000)
	count = count + 1
end

function test(inputs,target,caseNo)

	if testLosses == nil then
		testLosses = {}

	end

	print(string.rep("=",100))
	print(string.rep("=",100))
	print("Case number ".. caseNo..", with target output of ==>")
	--print(target)
	local averagePred = {}
	for i= 1, params.nTestPreds do
		outputs = model:forward(inputs[i])
		averagePred[i] = outputs
		loss = criterion:forward(outputs,target)
		--print(string.format("Case number %d with loss of %f",caseNo,loss))
	end
	local addTable = nn.CAddTable():cuda()
	averagePred = addTable:forward(averagePred)/params.nTestPreds
	averageLoss = criterion:forward(averagePred,target)
	print("==> Average prediction and loss")
	--print(averagePred)
	print(averageLoss)

	losses[count] = loss
	count = count + 1
	collectgarbage()
end


