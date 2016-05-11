Threads = require "threads"

do 
	local threadParams = params
	donkeys = Threads(
			params.nThreads,
			function(idx)
				params = threadParams
				require "torch"
				require "xlua"
				require "string"
				require "image"

				tid = idx -- Thread id
				print(string.format("Initialized thread %d of %d.", tid,params.nThreads))
				loadData = require "loadData"
				loadData.init(tid,params.nThreads)
			end
			)
end
	
