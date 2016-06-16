csv = require "csv"
require "paths"
local models = require "models"
local filterModel = torch.load("filter/filter.model"):cuda()
dofile("oneHotEncode.lua")

loadData = {}

function loadData.init(tid,nThreads,level)

	if tid == 1 or params.test ==1  then 

	elseif  params.test == 0 then
		print("==> Training")
		csvFile = "groundTruthTrain.csv"
		dataPath = "data/"
	else	
		print("Do not know train or test?")
	end

	local tableSplit
	local start
	local dataPath

        if params.test == 0 then 	
		if tid == 1 then
		-- If just tid = 1 thread is testing, the rest are training (i.e. full train)
			print(tid," ==> Testing")
			csvFile = "groundTruthTest.csv"
			dataPath = "data/"
			tableSplit = 1
			start = 1
		elseif tid ~=1 then
			print(tid," ==> Training")
			csvFile = "groundTruthTrain.csv"
			dataPath = "data/"
			start = tid - 1
			tableSplit = nThreads - 1
		end
	elseif params.test == 1 then
		print(tid," ==> Testing")
		csvFile = "groundTruthTest.csv"
		dataPath = "data/"
		-- if every thread is testing
		start = tid 
		tableSplit = nThreads 
	end

	local groundTruth = csv.csvToTable(dataPath .. csvFile) -- main truth table
	local obs = 1
	local nObs = csv.length(groundTruth)
	allPaths = {}

	count = 0
	for i = start, nObs, tableSplit do 

	        local row = groundTruth[i]:split(",")
	        local caseNumber, score, percScore = row[1], row[2], row[3]
		count = count + 1
	        local casePathHER2 = dataPath .. "roi_" .. caseNumber .. "/" .. level .."/" .. "HER2/"
	        local casePathHE = dataPath .. "roi_" .. caseNumber .. "/" .. level .."/" .. "HE/"
		--print(tid,caseNumber)

	        local imgPaths = {} 
	        local j = 1
		imgPaths.HER2 = {}
	        for f in paths.files(casePathHER2,".jpg") do
			 imgPaths.HER2[j] = casePathHER2 .. f
			 j = j + 1
	        end
		imgPaths.HE = {}
	        local j = 1
	        for f in paths.files(casePathHE,".jpg") do
			 imgPaths.HE[j] = casePathHE .. f
			 j = j + 1
	        end
	        allPaths[obs] = {imgPaths,score,percScore,caseNumber}
		obs = obs + 1
	 end 
	 --print("total count for thread number " .. tid .. "=", count)
	 collectgarbage()
end

function loadData.augmentCrop(img,windowSize)
	assert(windowSize<img:size(3),string.format("Window size %d is bigger than image size of %d.",
		windowSize,img:size(3)))
	-- Random  flips
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
	maxX = img:size(3) - windowSize
	maxY = img:size(2) - windowSize

	--- Try scaling ---
	return img:narrow(2,torch.random(maxY),windowSize):narrow(3,torch.random(maxX),windowSize) --cropped 
end

function loadData.loadXY(nWindows,windowSize)

	if params.test == 0 then
		-- Train
		currentTable = allPaths[torch.random(#allPaths)] -- Train is stochastic
	else
		-- Test
		if currentObs == nil then 
			currentObs = 1 
			nEpochs = 1
		elseif currentObs == #allPaths then 
			currentObs = 1
			nEpochs = nEpochs + 1
		else 
			currentObs = currentObs + 1	
		end
		currentTable = allPaths[currentObs] --Test is deterministic
	end


	local Xy = {}

	local allTensors = {}
	local imgDim
	function generateWindows(nWindows,HER2orHETable)
		local tensors = {}
		for i = 1, nWindows do
			local function suitableImg(HER2orHETable)
				 local suitablePic = false
				 local img
				 while suitablePic == false do 

					 local nObsT = csv.length(HER2orHETable)
					 local imgPath = HER2orHETable[torch.random(nObsT)] -- Draw random int to select window 
					 img = image.loadJPG(imgPath)
					 imgDim = img:size(3)
					 img = loadData.augmentCrop(img, windowSize)
					 local imgScale = image.scale(img,128,128,"simple"):cuda()
					 local output = filterModel:forward(imgScale:view(1,3,128,128))
					 if output[1] > 0.9 then suitablePic = true; end

				end
				return img:reshape(1,3,windowSize,windowSize)
			end
			tensors[i] = suitableImg(HER2orHETable) 
		end
		return torch.cat(tensors,1):cuda()
	end
	
	allTensors[1] = generateWindows(params.nHER2Windows,currentTable[1]["HER2"]) 
	--allTensors[2] = generateWindows(params.nHEWindows,currentTable[1]["HE"]) 
	Xy["data"] = allTensors[1] -- Just HER2 at the moment
	Xy["caseNo"] = currentTable[4] 
	Xy["score"] = currentTable[2] -- Normalize
	Xy["percScore"] = currentTable[3] -- Normalize

	--[[
	Xy["score"] = currentTable[2]/3 -- Normalize
	Xy["percScore"] = currentTable[3]/100 -- Normalize


	local target = torch.zeros(1,2)
	--local target = torch.zeros(params.nWindows + 1,2)
	target[{{},{1}}]:fill(Xy["score"])
	target[{{},{2}}]:fill(Xy["percScore"])
	target = target:cuda()
	Xy["target"] = target

	]]--
	Xy["target"] = oneHotEncode(Xy["score"],Xy["percScore"]):cuda()

	collectgarbage()
	return Xy, tid 
end

function loadData.main(display)
	require "cunn"

	models = require("resModels2")

	params = {}
	params.windowSize = 256 
	params.nHER2Windows= 5
	params.nHEWindows = 2 
	params.level = 2 
	params.nFeats = 16
	params.nLayers = 6 
	params.test = 0 
	params.nTestPreds = 10
	--model = models.resNetSiamese()

	if init == nil then
		require "image"
		loadData.init(2,2,params.level)
		init = "Not nil"
	end

	Xy = loadData.loadXY(params.nWindows,params.windowSize)

	function displayLoop()
		while true do 
			if dis == nil then
				initPic = torch.range(1,torch.pow(params.windowSize,2),1):reshape(params.windowSize,params.windowSize)
				imgDisplayHER2 = image.display{image=initPic, zoom=2, offscreen=false}
				imgDisplayHE = image.display{image=initPic, zoom=2, offscreen=false}
				image.display{image = Xy.data[1]:squeeze(), win = imgDisplayHER2, legend = "Score = ".. Xy["score"].. ". Case number = " .. Xy["caseNo"]}
				image.display{image = Xy.data[2]:squeeze(), win = imgDisplayHE, legend = "Score = ".. Xy["score"].. ". Case number = " .. Xy["caseNo"]}
				dis = 1
			end
			print(Xy)
			local title = string.format("Scores = {%f , %f} for case number %d",Xy["score"],Xy["percScore"],Xy["caseNo"])
			image.display{image = Xy.data[1]:squeeze(), win = imgDisplayHER2, legend = title}
			image.display{image = Xy.data[2]:squeeze(), win = imgDisplayHE, legend = title}
			Xy = loadData.loadXY(params.nWindows,params.windowSize)
			sys.sleep(2.5)
		end
	end
	if display == 1 then
		displayLoop()
	end
end


return loadData
