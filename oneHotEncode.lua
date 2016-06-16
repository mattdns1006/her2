-- Function to one hot encode
function oneHotEncode(score,percScore)
	local scoresTable = {}
	scoresTable["0"] = torch.Tensor{0,0,0}
	scoresTable["1"] = torch.Tensor{0,0,1}
	scoresTable["2"] = torch.Tensor{0,1,1}
	scoresTable["3"] = torch.Tensor{1,1,1}
	--[[
	percScoreTable = {}
	percScoreTable["0"] = torch.Tensor{0,0,0,0,0,0,0,0,0,0}
	percScoreTable["1"] = torch.Tensor{0,0,0,0,0,0,0,0,0,1}
	percScoreTable["2"] = torch.Tensor{0,0,0,0,0,0,0,0,1,1}
	percScoreTable["3"] = torch.Tensor{0,0,0,0,0,0,0,1,1,1}
	percScoreTable["4"] = torch.Tensor{0,0,0,0,0,0,1,1,1,1}
	percScoreTable["5"] = torch.Tensor{0,0,0,0,0,1,1,1,1,1}
	percScoreTable["6"] = torch.Tensor{0,0,0,0,1,1,1,1,1,1}
	percScoreTable["7"] = torch.Tensor{0,0,0,1,1,1,1,1,1,1}
	percScoreTable["8"] = torch.Tensor{0,0,1,1,1,1,1,1,1,1}
	percScoreTable["9"] = torch.Tensor{0,1,1,1,1,1,1,1,1,1}
	percScoreTable["10"]= torch.Tensor{1,1,1,1,1,1,1,1,1,1}
	]]--
	local oneHotScore = scoresTable[tostring(score)]
	local count = 1 
	local oneHotPercScore = torch.zeros(10)
	for i = 1,percScore, 10 do
		oneHotPercScore[{{-count}}] = 1
		count = count + 1
	end
	
	return torch.cat(oneHotScore,oneHotPercScore)
end

function oneHotDecode(prediction)
	local prediction = prediction:squeeze()
	local score = torch.sum(prediction[{{1,3}}])
	local percScore = torch.sum(prediction[{{4,-1}}])*10
	return score, percScore
end
