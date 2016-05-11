require "paths"

csv = {}

function csv.length(csv)
  	local count = 0
      	for _ in pairs(csv) do count = count + 1 end
        return count
end

function csv.csvToTable(path)
	local csvFile = io.open(path,"r")
	local header = csvFile:read()
	local data = {}

	local i = 0  
	for line in csvFile:lines('*l') do  
		i = i + 1
		data[i] = line 
	end
	return data
end

return csv 
