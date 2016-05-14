csv = require "csv"
require "paths"

loadData = {}

function loadData.init(tid,nThreads)


	local dataPath = "data/"
	local groundTruth = csv.csvToTable(dataPath .. "groundTruth.csv") -- main truth table
	allPaths = {}
	local nObs = csv.length(groundTruth)
	local obs = 1
	for i = tid, nObs, nThreads do 

	        local row = groundTruth[i]:split(",")
	        local caseNumber, score, percScore = row[1], row[2], row[3]
	        local casePath = dataPath .. "roi_" .. caseNumber .. "/"

	        local imgPaths = {} 
	        j = 1
	        for f in paths.files(casePath,".jpg") do
			 imgPaths[j] = casePath .. f
			 j = j + 1
	        end
	        allPaths[obs] = {imgPaths,score,percScore,caseNumber}
		obs = obs + 1
	 end 
	 collectgarbage()
end

function loadData.augmentCrop(img,windowSize)
	--angle = torch.uniform(2)
	--img = image.rotate(img,angle,"bilinear") 
	local hMid, wMid = img:size(2)/2, img:size(3)/2 -- middle
	local hStart, wStart = hMid - windowSize/2, wMid - windowSize/2
	
	return img:narrow(2,hStart,windowSize):narrow(3,wStart,windowSize) --cropped 
end

function loadData.loadXY(nWindows,windowSize)
	if currentObs == nil then currentObs = 1 end
	local currentTable = allPaths[currentObs]
	local nObsT = csv.length(currentTable[1])
	local tensors = {}
	local Xy = {}
	for i = 1, nWindows do
		 local imgPath = currentTable[1][torch.random(nObsT)] -- Draw random int to select window 
		 local img = image.loadJPG(imgPath)
		 local img = loadData.augmentCrop(img, windowSize)
		 tensors[i] = img:reshape(1,3,windowSize,windowSize)
	end
		
	Xy["data"] = torch.cat(tensors,1) 
	Xy["score"] = currentTable[2]/4
	Xy["percScore"] = currentTable[3]/100 -- Normalize
	Xy["caseNo"] = currentTable[4] 

	if currentObs == #allPaths then 
		currentObs = 1
	else 
		currentObs = currentObs + 1	
	end
	collectgarbage()
	return Xy 
end

function loadData.main()
	require "image"
	loadData.init(1,2)
	Xy = loadData.loadXY(10,896)
end

return loadData
