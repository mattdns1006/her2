Counter = {}
Counter.__index = Counter

function Counter.new()
	local self = {}
	return setmetatable(self,Counter)
end

function Counter:setContains(key)
	return self[key] ~= nil
end

function Counter:add(number)
	if self:setContains(tostring(number)) == false then
		self[tostring(number)] = 1
	else
		self[tostring(number)] = self[tostring(number)] + 1
	end
end
