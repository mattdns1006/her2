Threads = require "threads"

do 
	local threadParams = params
	donkeys = Threads(
			params.nThreads,
			function(idx)
				require "torch"
				require "xlua"
				require "string"
				tid = idx -- Thread id
				print(string.format("Initialized thread with id : %d.", tid))
				loadData = require "loadData"
				loadData.init(tid,threadParams.nThreads)
			end
			)
end
	
