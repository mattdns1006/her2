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

function ResultsTable:add (k,v)
	-- Checks to see if k exists in the table and adds v to mini table
	if self[k] == nil then
	   self[k] = {}
	   self[k][1] = v
	else
	   self[k][#self[k] + 1]  = v
	end
end

function ResultsTable:checkCount(number)
	for k,v in pairs(self) do
		local count = 0
		for x,y in ipairs(v) do
			count = count + 1
		end
		if count < number then
			return false
		end
	end
	return true

end

function ResultsTable:averagePred(number)
	averagePred = {}
	nPreds = number
	for k,v in pairs(self) do
		local t = {}
		for i =1, nPreds do
			t[i] = v[i][2]:double()
		end
		local t = torch.cat(t,1)
		local target = v[1][3]:mean(1):double()
		pair = {}
		pair["target"] = target
		pair["preds"] = t 
		averagePred[k] = pair	
	end
	return averagePred
end

function ResultsTable:averageLoss(number)
	local averageLosses = {}
	local averageLossesScore = {}
	local averageLossesPercScore = {}
	local nPreds = number 
	for k,v in pairs(self) do
		local t = {}
		for i =1, nPreds do
			t[i] = v[i][2]:double()
		end
		t = torch.cat(t,1)
		local mu = t:mean(1)
		local target = v[1][3]:mean(1):double()
		averageLosses[k] = criterion:double():forward(mu,target)
		averageLossesScore[k] = criterion:double():forward(mu[{{},{1}}],target[{{},{1}}])
		averageLossesPercScore[k] = criterion:double():forward(mu[{{},{2}}],target[{{},{2}}])
	end

	local function findMean(table)
		local tensor = torch.Tensor(tableLength(table))
		local i = 1
		for k, v in pairs(table) do
			tensor[i] = table[k]
			i = i + 1
		end
		return tensor:mean()
	end

	local losses = {}
	losses["invidual"] = {averageLosses,averageLossesScore,averageLossesPercScore}
	losses["average"] = {findMean(averageLosses),findMean(averageLossesScore),findMean(averageLossesPercScore)}
	collectgarbage()
	return losses 

end

function resultsEg()
	eg = ResultsTable.new()
	require "nn"
	criterion = nn.MSECriterion()
	x,y = 5 , 2
	nPreds = 2
	for i = 1, 3 do
		local targ = torch.uniform()
		for j = 1, nPreds do 
			eg:add(tostring(i), {1,torch.rand(x,y),torch.zeros(x,y):fill(targ)})
		end
	end
	losses = eg:averageLoss(nPreds,criterion)
	print(losses)
end


