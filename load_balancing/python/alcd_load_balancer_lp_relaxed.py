import pylpsparse as lps
import numpy as np
import time

class ALCDWrapper:
  # problem definition
  num_servers = 0
  num_shards = 0
  num_vars = 0
  max_memory = 0
  shard_loads = None
  shard_sizes = None
  current_locations = None
  replication_factor = 1

  # ALCD format variables
  primalA, primalb, primalc = None, None, None
  dualAt, dualb, dualc = None, None, None
  primalA_shape, dualAt_shape = None, None
  nb, nf, m, me = 0, 0, 0, 0
  decompressed = False

  def __init__(self, shard_loads, shard_sizes, current_locations, num_servers, max_memory):
    self.num_servers = num_servers
    self.num_shards = len(shard_loads)
    self.num_vars = self.num_servers * self.num_shards
    self.shard_loads = shard_loads
    self.shard_sizes = shard_sizes
    self.max_memory = max_memory
    self.current_locations = current_locations
    self.avg_load = sum(shard_loads) * 1.0 / num_servers
    self.epsilon_factor = 20
    self.constraint_scale = 50
  
  def __decompress(self):
    # 1. Create cost vector
    ## First num_vars ==> r vector, so no cost
    ## Second num_vars ==> x vector, so we have non-zero cost
    transfer_costs = np.zeros(2 * self.num_vars)
    for i in range(self.num_servers):
      for j in range(self.num_shards):
        val = 0 if self.current_locations[i][j] == 1 else self.shard_sizes[j]
        transfer_costs[self.num_vars + i * self.num_shards + j] = val

    # 2. Create constraint matrix
    mi = 2 * self.num_servers + self.num_servers + self.num_vars + self.num_shards
    me = self.num_shards
    Amat = lps.Matrix(mi + me + 1)
    bvec = np.zeros(mi + me + 1)
    row_offset = 0
    ### Load constraints [on r values] --> Inequality
    min_load, max_load = - self.avg_load/self.epsilon_factor, self.avg_load/self.epsilon_factor
    for i in range(self.num_servers):
      nzidxs = [i * self.num_shards + j for j in range(self.num_shards)]
      nzvals = [x - self.avg_load for x in self.shard_loads]
      nzvals2 = [-x for x in nzvals]
      # load on server i <= avg_load + epsilon
      Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
      bvec[row_offset] = max_load
      # load on server i >= avg_load - epsilon
      Amat.setrow(row_offset + 1, list(zip(nzidxs, nzvals2)))
      bvec[row_offset + 1] = -min_load
      row_offset += 2
    ### Memory constraints [on x values] --> Inequality
    for i in range(self.num_servers):
      nzidxs = [self.num_vars + i * self.num_shards + j for j in range(self.num_shards)]
      nzvals = self.shard_sizes
      # memory on server i <= max_memory
      Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
      bvec[row_offset] = self.max_memory
      row_offset += 1
    ### Set r <= x  --> Inequality
    for i in range(self.num_servers):
      for j in range(self.num_shards):
        nzidxs = [i * self.num_shards + j, self.num_vars + i * self.num_shards + j]
        nzvals = [self.constraint_scale, -1 * self.constraint_scale]
        Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
        bvec[row_offset] = 0
        row_offset += 1
    ### Replication factor constraints --> Inequality
    for j in range(self.num_shards):
      nzidxs = [i * self.num_shards + j for i in range(self.num_servers)]
      nzvals = [-1 * self.constraint_scale] * self.num_servers
      Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
      bvec[row_offset] = -1 * self.constraint_scale * self.replication_factor
      row_offset += 1
    assert row_offset == mi, f"Row offset mismatch: {row_offset} != {mi}"
    ### Sum (r) = 1 --> Equality
    for j in range(self.num_shards):
      nzidxs = [i * self.num_shards + j for i in range(self.num_servers)]
      nzvals = [self.constraint_scale] * self.num_servers
      Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
      bvec[row_offset] = self.constraint_scale
      row_offset += 1
    assert row_offset == mi + me, f"Row offset mismatch: {row_offset} != {mi + me}"

    # 3. Create primal and dual matrices
    self.primalA = Amat
    self.primalA_shape = (mi + me, 2 * self.num_vars)
    self.primalb = bvec
    self.primalc = transfer_costs
    self.dualAt = Amat.transpose()
    self.dualAt_shape = (2 * self.num_vars, mi + me)
    self.dualb = transfer_costs
    self.dualc = bvec
    self.nb, self.nf = 2 * self.num_vars, 0
    self.m, self.me = mi, me
    self.decompressed = True

  def get_primal_alcd_format(self):
    if not self.decompressed:
      self.__decompress()
    return (self.primalA, self.primalb, self.primalc, self.nb, self.nf, self.m, self.me)
  
  def get_dual_alcd_format(self):
    if not self.decompressed:
      self.__decompress()
    ## TODO (suhasjs) --> Check if nb, nf, m, me are correctly returned???
    return (self.dualAt, self.dualb, self.dualc, self.me, self.m, self.nb, self.nf)

def balance_load_alcd(num_shards, num_servers, shard_loads, shard_memory_usages,
                                     current_locations, sample_queries, max_memory):
  # Create ALCDWrapper object
  load_start_t = time.time()
  print(f"Creating ALCD format from inputs")
  lpobj = ALCDWrapper(shard_loads, shard_memory_usages, current_locations, num_servers, max_memory)
  primal_args = lpobj.get_primal_alcd_format()
  dual_lpargs = lpobj.get_dual_alcd_format()
  load_end_t = time.time()

  # Create args for ALCD solver
  lpcfg = lps.LP_Param()
  lpcfg.solve_from_dual = False
  lpcfg.eta = 1
  lpcfg.verbose = True
  lpcfg.tol_trans = 1e-1
  lpcfg.tol = 1e-3
  # lpcfg.tol_sub = args.alcd_tol
  lpcfg.tol_sub = 1e-1
  lpcfg.use_CG = False
  lpcfg.max_iter = 100
  lpcfg.inner_max_iter = 100

  # Initialize ALCD solver
  A, b, c, nb, nf, m, me = primal_args
  At = dual_lpargs[0]
  x0 = np.zeros(len(c))
  w0 = np.ones(len(b))
  init_start_time = time.time()
  print(f"Initalizing ALCD solver")
  if lpcfg.solve_from_dual is False:
    h2jj = np.zeros(nb + nf)
    hjj_ubound = np.zeros(nb + nf)
    lps.init_state(x0, w0, h2jj, hjj_ubound, nb, nf, m, me, A, b, c, lpcfg.eta) 
  else:
    h2jj = np.zeros(m + me)
    hjj_ubound = np.zeros(m + me)
    lps.init_state(w0, x0, h2jj, hjj_ubound, m, me, nb, nf, At, c, b, lpcfg.eta)
  init_end_time = time.time()

  # Solve via ALCD solver
  x0[:] = 0
  w0[:] = 1
    
  # Solve using ALCD
  lpinfo = lps.LP_Info()
  solve_start_time = time.time()
  print(f"Solving using ALCD solver")
  # lps.solve_alcd(A, b, c, x0, w0, h2jj, hjj_ubound, nb, nf, m, me, lpcfg, lpinfo)
  lps.solve_alcd(A, b, c, x0, w0, h2jj, hjj_ubound, nb, nf, m, me, lpcfg, lpinfo)
  solve_end_time = time.time()

  # Extract results
  rvars = x0[:num_shards*num_servers].round(3)
  xvars = x0[num_shards*num_servers:].round(3)
  # convert to list
  rvars = rvars.reshape((num_servers, num_shards)).tolist()
  xvars = xvars.reshape((num_servers, num_shards)).tolist()
  print(f"ALCD Solver: Load={(load_end_t - load_start_t)*1000:.1f}ms, Init: {(init_end_time - init_start_time)*1000:.1f}ms, Solve time: {(solve_end_time - solve_start_time)*1000:.1f}ms")
  return rvars, xvars