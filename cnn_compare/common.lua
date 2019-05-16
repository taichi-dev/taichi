local paths = require('paths')
local lfs = require('lfs')

local common = {}


package.path = '../../th/?/init.lua;' .. package.path

function common.walk_paths(root, ext)
  local function walk_paths_(d, ps) 
    for path in paths.iterdirs(d) do
      local path = paths.concat(d, path)
      walk_paths_(path, ps)
    end
    for path in paths.iterfiles(d) do
      local path = paths.concat(d, path)

      if ext then
        if ext == paths.extname(path) then
          table.insert(ps, path)
        end
      else 
        table.insert(ps, path)
      end
    end
  end 

  local ps = {}
  walk_paths_(root, ps)
  table.sort(ps)
  return ps
end

function common.paths_from_file(path)
  local ps = {}
  for p in io.lines(path) do
    table.insert(ps, p)
  end
  return ps
end 

function common.walk_paths_cached(root, ext)
  local cache_path 
  if ext then 
    cache_path = paths.concat(root, string.format('cache_paths_%s.txt', ext))
  else
    cache_path = paths.concat(root, string.format('cache_paths.txt'))
  end

  if paths.filep(cache_path) then
    print('[INFO] load paths from cache '..cache_path)
    return common.paths_from_file(cache_path)
  else
    print('[INFO] write paths to cache '..cache_path)
    local ps = common.walk_paths(root, ext)
    local f = io.open(cache_path, 'w')
    for _, p in ipairs(ps) do
      f:write(string.format('%s\n', p))
    end
    io.close(f)
    return ps
  end
end

function common.match_paths(ps, pattern)
  local rps = {}
  for _, p in ipairs(ps) do
    if p:match(pattern) then
      table.insert(rps, p)
    end
  end 
  table.sort(rps)
  return rps
end 

function common.table_shuffle(tab)
  local cnt = #tab
  while cnt > 1 do 
    local idx = math.random(cnt)
    tab[idx], tab[cnt] = tab[cnt], tab[idx]
    cnt = cnt - 1
  end
end 

function common.table_length(tab)
  local cnt = 0
  for _ in pairs(tab) do cnt = cnt + 1 end
  return cnt
end 

function common.table_clear(tab)
  for k in pairs(tab) do tab[k] = nil end
end

function common.table_combine(tab1, tab2)
  local tab_c = {}
  for idx = 1, #tab1 do
    table.insert(tab_c, tab1[idx])
  end
  for idx = 1, #tab2 do
    table.insert(tab_c, tab2[idx])
  end
  return tab_c
end 

function common.string_split(str, sSeparator, nMax, bRegexp)
  assert(sSeparator ~= '')
  assert(nMax == nil or nMax >= 1)
  local aRecord = {}

  if str:len() > 0 then
    local bPlain = not bRegexp
    nMax = nMax or -1

    local nField, nStart = 1, 1
    local nFirst,nLast = str:find(sSeparator, nStart, bPlain)
    while nFirst and nMax ~= 0 do
      aRecord[nField] = str:sub(nStart, nFirst-1)
      nField = nField+1
      nStart = nLast+1
      nFirst,nLast = str:find(sSeparator, nStart, bPlain)
      nMax = nMax-1
    end
    aRecord[nField] = str:sub(nStart)
  end
  return aRecord
end


function common.net_he_init(net)
  local function conv_init(model, name)
    for k,v in pairs(model:findModules(name)) do
      local n = v.kT * v.kW * v.kH * v.nOutputPlane
      v.weight:normal(0, math.sqrt(2/n))
      v.bias:zero()
    end 
  end 

  local function linear_init(model)
    for k, v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
    end
  end 

  print('[INFO] INIT NET ALA HE')
  conv_init(net, 'cudnn.VolumetricConvolution')
  conv_init(net, 'nn.VolumetricConvolution')
  conv_init(net, 'oc.OctreeConvolutionMM')
  linear_init(net)
end


function common.train_epoch(opt, data_loader)
  local net = opt.net or error('no net in train_epoch')
  local criterion = opt.criterion or error('no criterion in train_epoch')
  local optimizer = opt.optimizer or error('no optimizer in train_epoch')
  local n_batches = data_loader:n_batches()

  net:training()

  local parameters, grad_parameters = net:getParameters()
  for batch_idx = 1, n_batches do
    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      grad_parameters:zero()

      local input, target = data_loader:getBatch()

      local output = net:forward(input)
      local f = criterion:forward(output, target)
      local dfdx = criterion:backward(output, target)
      net:backward(input, dfdx)
      
      if batch_idx < 129 or batch_idx % math.floor((n_batches / 200)) == 0 then 
        print(string.format('epoch=%2d | iter=%4d | loss=%9.6f ', opt.epoch, batch_idx, f))
      end
      
      return f, grad_parameters
    end 
    optimizer(feval, parameters, opt)
    xlua.progress(batch_idx, n_batches)
  end 
end

function common.test_epoch(opt, data_loader)
  local net = opt.net or error('no net in test_epoch')
  local criterion = opt.criterion or error('no criterion in test_epoch')
  local n_batches = data_loader:n_batches()

  net:evaluate()

  local avg_f = 0
  local accuracy = 0
  local n_samples = 0
  for batch_idx = 1, n_batches do
    print(string.format('[INFO] test batch %d/%d', batch_idx, n_batches))

    local timer = torch.Timer()
    local input, target = data_loader:getBatch()
    print(string.format('[INFO] loading data took %f[s] - n_batches %d', timer:time().real, target:size(1)))

    local timer = torch.Timer()
    local output = net:forward(input)
    output = output[{{1,target:size(1)}, {}}]
    local f = criterion:forward(output, target)
    print(string.format('[INFO] net/crtrn fwd took %f[s]', timer:time().real))
    avg_f = avg_f + f
    
    local maxs, indices = torch.max(output, 2)
    for bidx = 1, target:size(1) do
      if indices[bidx][1] == target[bidx] then
        accuracy = accuracy + 1
      end
      n_samples = n_samples + 1
    end
  end 
  avg_f = avg_f / n_batches
  accuracy = accuracy / n_samples

  print(string.format('test_epoch=%d, avg_f=%f, accuracy=%f', opt.epoch, avg_f, accuracy))
end


function common.worker(opt, train_data_loader, test_data_loader)

  -- enable logging
  local cmd = torch.CmdLine()
  cmd:log(paths.concat(opt.out_root, string.format('train_%d.log', sys.clock())))

  -- load state if it exists
  local state_path = paths.concat(opt.out_root, 'state.t7')
  if paths.filep(state_path) then 
    print('[INFO] LOADING PREVIOUS STATE')
    local opt_state = torch.load(state_path)
    opt_state.do_stats = opt.do_stats
    opt_state.save_output = opt.save_output
    for k, v in pairs(opt_state) do 
      if k ~= 'criterion' then
        opt[k] = v
      end
    end
  end

  local start_epoch = 1
  if opt.epoch then
    start_epoch = opt.epoch + 1
  end
  print(string.format('[INFO] start_epoch=%d', start_epoch))
  for epoch = start_epoch, opt.n_epochs do
    opt.epoch = epoch
    
    -- clean up
    opt.net:clearState()
    collectgarbage('collect')
    collectgarbage('collect')

    -- train
    print('[INFO] train epoch '..epoch..', lr='..opt.learningRate)
    opt.data_fcn = opt.train_data_fcn
    common.train_epoch(opt, train_data_loader)
     
    -- save network
    print('[INFO] save net')
    local net_path = paths.concat(opt.out_root, string.format('net_epoch%03d.t7', opt.epoch))
    torch.save(net_path, opt.net:clearState())
    print('[INFO] saved net to: ' .. net_path)

    -- save state
    if not opt.state_save_interval or opt.epoch % opt.state_save_interval == 0 then
      print('[INFO] save state')
      opt.net = opt.net:clearState()
      torch.save(state_path, opt)
      print('[INFO] saved state to: ' .. state_path)
    end

    -- clean up 
    collectgarbage('collect')
    collectgarbage('collect')

    -- adjust learning rate
    if opt.learningRate_steps[epoch] ~= nil then
      opt.learningRate = opt.learningRate * opt.learningRate_steps[epoch]
    end
  end
    
  -- test network
  common.test_epoch(opt, test_data_loader)
end

return common
