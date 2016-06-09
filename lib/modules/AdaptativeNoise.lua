require 'nn'
require 'nngraph'
require 'NLPRelationClassification/lib/modules/GaussianSampler'
if(not ADAPTATIVENOISE) then
  ADAPTATIVENOISE = true
  local AdaptativeNoise, parent = torch.class('nn.AdaptativeNoise', 'nn.Module')
  function AdaptativeNoise:__init()
    parent.__init(self)
    self.gradInput = {}
    self.gaussianSampler = nn.GaussianSampler()
    self.add             = nn.CAddTable()
    self.l = true
  end
  function AdaptativeNoise:updateOutput(input)
    if(self.train ~= false) then
      self.gaussianSamplerOutput = self.gaussianSampler:updateOutput({input[2],input[3]})
      self.output = self.add:updateOutput({input[1],self.gaussianSamplerOutput})
    else
      self.output = input[1]
    end
    return self.output
  end
  function AdaptativeNoise:updateGradInput(input, gradOutput)
    if(self.train ~=false) then
      self.gradInputAdd = self.Add:updateGradInput({input[1],self.gaussianSamplerOutput},gradOutput)
      self.gradInput    = self.gaussianSampler:updateGradInput({input[2],input[3]},self.gradInputAdd)
    else
      self.gradInput = {gradOutput,torch.zeros(input[2]:size()),torch.zeros(input[3]:size())}
    end
    return self.output
  end

end
