require "paths"
require "image"
require "cunn"

local function fileExists(name)
	local f=io.open(name,"r")
	if f~=nil then io.close(f) return true else return false end
end

loadData = {}

loadData.augPaths = {}
loadData.trainPaths = {}
loadData.testPaths = {}
local path = "data/augmented/"
local level = tostring(7) 
for f in paths.files(path,"x") do
	local folder = path .. f 
	loadData.augPaths[#loadData.augPaths+1] = folder
end

local path = "data/"
for f in paths.files(path,"roi_") do
	local folder = path .. f .. "/".. level .. "/"
	for pic in paths.files(folder,"HER2") do
			loadData.trainPaths[#loadData.trainPaths+1] = folder..pic
	end
end
local path = "testData/"
for f in paths.files(path,"roi_") do
	local folder = path .. f .. "/".. level .. "/"
	for pic in paths.files(folder,"HER2") do
			loadData.testPaths[#loadData.testPaths+1] = folder..pic
	end
end

function loadData.loadObs(trainOrTest)
	local x, y
	local rObs
	if trainOrTest == "aug" then
		rObs = loadData.augPaths[torch.random(#loadData.trainPaths)]
		x,y = image.loadJPG(rObs), image.loadJPG(rObs:gsub("x","y"))

		local randInt = torch.random(4)
		if randInt == 1 then	
		elseif randInt == 2 then
			image.vflip(x,x)
			image.vflip(y,y)
		elseif randInt == 3 then
			image.hflip(x,x)
			image.hflip(y,y)
		elseif randInt == 4 then
			image.vflip(x,x)
			image.vflip(y,y)
			image.hflip(x,x)
			image.hflip(y,y)
		end
	elseif trainOrTest == "train" then
		rObs = loadData.trainPaths[torch.random(#loadData.trainPaths)]
		x,y = image.loadJPG(rObs), image.loadJPG(rObs:gsub("HER2.jpg","HE.jpg"))
	else 
		rObs = loadData.testPaths[torch.random(#loadData.testPaths)]
		x,y = image.loadJPG(rObs), image.loadJPG(rObs:gsub("HER2.jpg","HE.jpg"))
	end

	-- Random flips
	return x:cuda():resize(1,x:size(1),x:size(2),x:size(3)),y:cuda():resize(1,y:size(1),y:size(2))
end

return loadData
