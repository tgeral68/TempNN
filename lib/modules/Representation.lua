require 'nn'
require 'nngraph'
if(not REPRESENTATIONCLASS) then
	REPRESENTATIONCLASS = true
	local Representation, parent = torch.class('nn.Representation', 'nn.LookupTable')
	function Representation:__init(a,b)
		parent.__init(self,a,b)
		self.indexChange  = {}
		self.narrowTableGW = {}
		self.hasUpdate = false
		self.mask = torch.ByteTensor(self.weight:size()):fill(1)
	end

	function Representation:updateIndexes(indexes)
		self.hasUpdate = true
		for i = 1, indexes:size(1) do
			self.mask[indexes[i]] = 0
		end
	end

	function Representation:updateOutput(input)
		parent.updateOutput(self,input)
		return  self.output
	end
	function Representation:accGradParameters(input, gradOutput, scale)
		if(self.hasUpdate) then
			parent.accGradParameters(self,input, gradOutput, scale)
			self.gradWeight:maskedFill(self.mask,0.)
		end
	end
	function Representation:addUpgradeWeight(index)
		self.mask[index]:fill(0)
 	end



	function Representation:saveRepresentation(dict,filepath)
		local file = io.open(filepath,"w")
		local w = self.weight
		for i=1, w:size(1) do
			str = dict[i]
			if(str~=nil) then
				for j = 1 , w:size(2) do
					str = str.." "..w[i][j]
				end
			file:write(str.."\n") end
		end
		file:close()
	end
end
