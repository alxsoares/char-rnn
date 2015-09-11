
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Score sequence using a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-primetextfile',"",'file with the lines to score')
cmd:option('-verbose',0,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
gprint('creating an LSTM...')
local current_state
local num_layers = checkpoint.opt.num_layers
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(current_state, h_init:clone())
    table.insert(current_state, h_init:clone())
end
state_size = #current_state


local f = io.open(opt.primetextfile, 'r')
while true do
    local seed_text = f:read("*line")
    if seed_text==nil then break end
    -- start with uniform probabilities
    prediction = torch.log(torch.Tensor(1, #ivocab):fill(1)/(#ivocab))
    if opt.gpuid >= 0 then prediction = prediction:cuda() end
    totalLogProb = 0
    if string.len(seed_text) > 0 then
        gprint('seeding with ' .. seed_text)
        gprint('--------------------------')
        for c in seed_text:gmatch'.' do
            prev_char = torch.Tensor{vocab[c]}
            prev_char_prob = prediction[1][vocab[c]]
            totalLogProb = totalLogProb + prev_char_prob
            --io.write(ivocab[prev_char[1]])
            if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
            local lst = protos.rnn:forward{prev_char, unpack(current_state)}
            -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
            current_state = {}
            for i=1,state_size do table.insert(current_state, lst[i]) end
            prediction = lst[#lst] -- last element holds the log probabilities
        end
    end

    --print("log prob",totalLogProb)
    io.write(totalLogProb);
    io.write('\n');
    io.flush();
    --io.write('\n') io.flush()
end
