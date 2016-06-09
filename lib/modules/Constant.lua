require 'nn'
------------------------------------------------------------------------
--[[ Constant ]]--
-- Outputs a constant value given an input.
-- If nInputDim is specified, uses the input to determine the size of
-- the batch. The value is then replicated over the batch.
-- You can use this with nn.ConcatTable() to append constant inputs to
-- an input : nn.ConcatTable():add(nn.Constant(v)):add(nn.Identity()) .
------------------------------------------------------------------------
if(not CONSTANTCLASS) then
  CONSTANTCLASS = true
  local Constant, parent = torch.class("nn.Constant", "nn.Module")
  function Constant:__init(value, outputDims, inputDims)
   parent.__init(self)
   self.inputDims = inputDims or 1
   self.outputDims= outputDims
   self.value = value
   self.output = torch.Tensor(outputDims):fill(value)
  end
  function Constant:updateOutput(input)
   if (input==nil) then self.output:resize(self.outputDims)
   elseif input:dim() == self.inputDims then
      self.output:resize(self.outputDims)
   else
      self.output:resize(input:size(1),self.outputDims):fill(self.value)
   end
   return self.output
  end
  function Constant:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   return self.gradInput
  end

  function Constant:cuda()
     self.output=self.output:cuda()
     self.gradInput=self.gradInput:cuda()
  end
end
