csv = require "csv"
require "paths"

loadData = {}

function loadData.init(tid,nThreads)

	-- Main truth table 
	dataPath = "data/"
	groundTruth = csv.csvToTable(dataPath .. "groundTruth.csv")
	allPaths = {}
	nObs = csv.length(groundTruth)
	for i = tid, nObs, nThreads do 

	        local row = groundTruth[i]:split(",")
	        local caseNumber, score, percScore = row[1], row[2], row[3]
	        local casePath = dataPath .. "roi_" .. caseNumber .. "/"

	        imgPaths = {} 
	        j = 1
	        for f in paths.files(casePath,".jpg") do
			 imgPaths[j] = casePath .. f
			 j = j + 1
	        end
	        allPaths[i] = imgPaths 
	 end 
end

function loadData.loadXY()
	 
end

return loadData
