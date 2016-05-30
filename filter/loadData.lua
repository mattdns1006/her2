dofile("../csv.lua")
require "paths"
require "image"

local csvs = {}
csvs[1] = csv.csvToTable("filterDataLabelledCSV.csv")
csvs[2] = csv.csvToTable("filterDataLabelledExtraCSV.csv")
filePaths0 = {}
filePaths1 = {}

for csvName, csvPath in ipairs(csvs) do
	for k, v in ipairs(csvPath) do 
		obs = v:split(",")
		y = obs[3]
		if tonumber(y) == 0 then
			filePaths0[#filePaths0+1] = v
		else
			filePaths1[#filePaths1+1] = v 
		end
	end
end


loadData = {}

function loadData.loadObs()
	local rObs
	if torch.uniform() < 0.5 then
		rObs = filePaths0[torch.random(#filePaths0)]:split(",")
	else
		rObs = filePaths1[torch.random(#filePaths1)]:split(",")
	end
	local xPath,y = rObs[2],tonumber(rObs[3])
	xPath = xPath:gsub("data/","data/labelled/")
	local img = image.loadJPG(xPath)
	-- Random flips
	local randInt = torch.random(4)
	if randInt == 1 then	
	elseif randInt == 2 then
		image.vflip(img,img)
	elseif randInt == 3 then
		image.hflip(img,img)
	elseif randInt == 4 then
		image.vflip(img,img)
		image.hflip(img,img)
	end
	img:resize(1,3,params.windowSize,params.windowSize)
	return img,y
end

return loadData
