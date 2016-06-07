require "paths"
require "image"
require "cunn"

local function fileExists(name)
	local f=io.open(name,"r")
	if f~=nil then io.close(f) return true else return false end
end

local loadData = {}

loadData.trainPaths = {}
loadData.testPaths = {}
local path = "data/"
local level = tostring(7) 
local i = 1
for f in paths.files(path,"roi_") do
	local folder = path .. f .. "/".. level .. "/"
	for pic in paths.files(folder,"HER2") do
		if i < 36 then 
			loadData.trainPaths[#loadData.trainPaths+1] = folder..pic
		else
			loadData.testPaths[#loadData.testPaths+1] = folder..pic
		end
	end
	i = i + 1
end


function loadData.loadObs(dataPaths)
	rObs = dataPaths[torch.random(#dataPaths)]
	x,y = image.loadJPG(rObs), image.loadPNG(rObs:gsub("HER2.jpg","y1.png"))

	-- Random flips
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
	return x:cuda():resize(1,x:size(1),x:size(2),x:size(3)),y:cuda():resize(1,y:size(1),y:size(2))
end

return loadData
