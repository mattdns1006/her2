ResultsTable = {}
ResultsTable.__index = ResultsTable

function tableLength(t)
  local count = 0
    for _ in pairs(t) do count = count + 1 end
      return count
end

function ResultsTable.new()
	local self = {}
	return setmetatable(self,ResultsTable)
end

function ResultsTable:add (x,y,target)
	local k = tostring(x)
	-- Checks to see if k exists in the table and adds v to mini table
	if self[k] == nil then
	   self[k] = {}
	   self[k]["target"] = target:double()
	   self[k]["predictions"] = {} 
	   self[k]["predictions"][#self[k]["predictions"] + 1] = y:double()
	else
	   self[k]["predictions"][#self[k]["predictions"] + 1]  = y:double()
	end
end

function ResultsTable:checkCount(number)
	for k,v in pairs(self) do
		local count = 0
		for x,y in ipairs(v["predictions"]) do
			count = count + 1
		end
		if count < number then
			return false
		end
	end
	return true
end

function ResultsTable:averagePrediction(meanOrMedian)
	--assert(self:checkCount(number)==true,"Need to have at least "..number.." predictions to get average of " .. number ".")
	local overall = {} 
	local overallScoreLoss = {}
	local overallPercScoreLoss = {}
	for k, v in pairs(self) do
		local predictions = torch.cat(v["predictions"],1)
		self[k]["predictionsT"] = predictions
		local predMu
		if meanOrMedian == "mean" then 
			predMu = predictions:mean(1)
		else
			predMu = predictions:median(1)
		end
		local predMuScore, predMuPercScore = predMu[{{},{1}}],predMu[{{},{2}}]
		self[k]["meanPrediction"] = predMu
		local target = self[k]["target"]
		local targetScore, targetPredScore = target[{{},{1}}], target[{{},{2}}]
		self[k]["meanLoss"] = {}
		self[k]["meanLoss"][1] = criterion:forward(predMu,target)
		self[k]["meanLoss"][2] = criterion:forward(predMuScore,targetScore)
		self[k]["meanLoss"][3] = criterion:forward(predMuPercScore,targetPredScore)

		overall[#overall + 1] = self[k]["meanLoss"][1]
		overallScoreLoss[#overallScoreLoss+ 1] = self[k]["meanLoss"][2]
		overallPercScoreLoss[#overallPercScoreLoss+ 1] = self[k]["meanLoss"][3]

	end
	overall = torch.Tensor(overall):mean()
	overallScoreLoss = torch.Tensor(overallScoreLoss):mean()
	overallPercScoreLoss = torch.Tensor(overallPercScoreLoss):mean()
	return {overall,overallScoreLoss,overallPercScoreLoss} 
end


function resultsEg()

	eg = ResultsTable.new()
	require "nn"
	require "cunn"
	m = torch.load("models/resNet1.model_3_20_10_128_12_30000")
	criterion = nn.MSECriterion():cuda()
	params = {}
	params.windowSize = 128
	params.nHER2Windows = 20
	params.nHEWindows = 10

	for caseNumber = 1, 10 do -- 10 test examples
		y = torch.Tensor{torch.uniform(),torch.uniform()}:reshape(1,2):cuda() --constant
		for j =1, 3 do -- 3 predictions per example
			local x1 = torch.rand(params.nHER2Windows,3,params.windowSize,params.windowSize):cuda()
			local x2 = torch.rand(params.nHEWindows,3,params.windowSize,params.windowSize):cuda()
			X = {x1,x2}

			yPred = m:forward(X)
			eg:add(caseNumber,yPred,y)
		end
	end
end


