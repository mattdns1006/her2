csv = require "csv"
require "paths"
local models = require "models"
local filterModel = torch.load("filter/filter.model"):cuda()

loadData = {}

function loadData.init(tid,nThreads,level)

	local dataPath
	if params.test == 0 then 
		print("==> Training")
		csvFile = "groundTruthTrain.csv"
		dataPath = "data/"
	elseif params.test == 1 and params.actualTest ==0 then
		print("==> Testing")
		csvFile = "groundTruthTest.csv"
		dataPath = "data/"
	else	
		print("==> True test")
		csvFile = "groundTruth.csv"
		dataPath = "testData/"
	end

	local groundTruth = csv.csvToTable(dataPath .. csvFile) -- main truth table
	allPaths = {}
	local nObs = csv.length(groundTruth)
	local obs = 1
	for i = tid, nObs, nThreads do 

	        local row = groundTruth[i]:split(",")
	        local caseNumber, score, percScore = row[1], row[2], row[3]
	        local casePathHER2 = dataPath .. "roi_" .. caseNumber .. "/" .. level .."/" .. "HER2/"
	        local casePathHE = dataPath .. "roi_" .. caseNumber .. "/" .. level .."/" .. "HE/"

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

	local tensors = {}
	tensors[1] = {}
	tensors[2] = {}
	local imgDim
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
		tensors[1][i] = suitableImg(currentTable[1].HER2) 
		tensors[2][i] = suitableImg(currentTable[1].HE)

	end
	
	tensors[1] = torch.cat(tensors[1],1):cuda() 
	tensors[2] = torch.cat(tensors[2],1):cuda()
	Xy["data"] = tensors 
	Xy["score"] = currentTable[2]/3 -- Normalize
	Xy["percScore"] = currentTable[3]/100 -- Normalize
	Xy["caseNo"] = currentTable[4] 
	--Xy["coverage"] = params.nWindows*(torch.pow(params.windowSize,2)/torch.pow(imgDim,2))/#currentTable[1]

	local target = torch.zeros(1,2)
	--local target = torch.zeros(params.nWindows + 1,2)
	target[{{},{1}}]:fill(Xy["score"])
	target[{{},{2}}]:fill(Xy["percScore"])
	target = target:cuda()
	Xy["target"] = target

	collectgarbage()
	return Xy 
end

function loadData.main(display)
	require "cunn"

	models = require("resModels2")
	model = models.resNetSiamese()
	params = {}
	params.windowSize = 256 
	params.nWindows = 5
	params.level = 2 
	params.nFeats = 16
	params.nLayers = 6 
	params.test = 0
	params.nTestPreds = 10

	if init == nil then
		require "image"
		loadData.init(1,1,params.level)
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
