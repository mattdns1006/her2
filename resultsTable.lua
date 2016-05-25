ResultsTable = {}
ResultsTable.__index = ResultsTable

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


