require "gnuplot"
dofile("round.lua")

ConfusionMatrix = {}
ConfusionMatrix.__index = ConfusionMatrix

function ConfusionMatrix.new(nClasses)
	local self = {}
	self.cm = torch.zeros(nClasses,nClasses)
	return setmetatable(self,ConfusionMatrix)
end

function ConfusionMatrix:allocate(matrix,prediction,target)
	matrix[round(prediction)+1][target+1] = 1 + matrix[round(prediction)+1][target+1]
end

function ConfusionMatrix:add(prediction,target)
	assert(prediction and type(prediction) == 'number', "Prediction must be a number")
	assert(target and type(target) == 'number',"Target must be a number")
	self:allocate(self.cm,prediction,target,self.threshold) -- add to main Confusion matrix
end

function ConfusionMatrix:performance()
	self.accuracy = torch.diag(self.cm):sum()/self.cm:sum()
	--self.precision = (self.cm[2][2])/(self.cm[2][1] + self.cm[2][2])
	--self.recall = (self.cm[2][2])/(self.cm[1][2] + self.cm[2][2])
	print(string.format("Accuracy %f.", self.accuracy))
end

function ConfusionMatrix:reset()
	self.cm:fill(0)
end

--Example
function cmEg()
	cm = ConfusionMatrix.new(4)
	for i=1, 1000 do
		target = torch.random(3)
		pred = torch.randn(1) + target 
		pred:clamp(0,3)
		cm:add(round(pred:squeeze()),target)
	end
end
--cmEg()
