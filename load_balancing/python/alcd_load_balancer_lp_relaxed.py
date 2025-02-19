import pylpsparse as lps
import numpy as np
from scipy.sparse import coo_matrix
import time
import cvxpy as cp

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
  only_inequalities = False

  # ALCD format variables
  primalA, primalb, primalc = None, None, None
  dualAt, dualb, dualc = None, None, None
  primalA_shape, dualAt_shape = None, None
  nb, nf, m, me = 0, 0, 0, 0
  decompressed = False

  def __init__(self, shard_loads, shard_sizes, current_locations, num_servers, max_memory, only_inequalities):
    self.num_servers = num_servers
    self.num_shards = len(shard_loads)
    self.num_vars = self.num_servers * self.num_shards
    self.shard_loads = shard_loads
    self.shard_sizes = shard_sizes
    self.max_memory = max_memory
    self.only_inequalities = only_inequalities
    self.current_locations = current_locations
    self.avg_load = sum(shard_loads) * 1.0 / num_servers
    self.epsilon_factor = 20
    self.constraint_scale = 10
    self.Acoo = None
  
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
    coo_data, coo_row, coo_col = [], [], []
    if self.only_inequalities:
      mi = 2 * self.num_servers + self.num_servers + self.num_vars + self.num_shards + 2 * self.num_shards
      me = 0
    else:
      mi = 2 * self.num_servers + self.num_servers + self.num_vars + self.num_shards
      me = self.num_shards
    Amat = lps.Matrix(mi + me)
    bvec = np.zeros(mi + me)
    row_offset = 0
    ### Load constraints [on r values] --> Inequality
    max_load_diff = self.avg_load / self.epsilon_factor
    for i in range(self.num_servers):
      nzidxs = [i * self.num_shards + j for j in range(self.num_shards)]
      # nzvals = [x - self.avg_load for x in self.shard_loads]
      # nzvals2 = [-x for x in nzvals]
      nzvals = [x for x in self.shard_loads]
      nzvals2 = [-x for x in nzvals]
      # load on server i <= avg_load + epsilon
      Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
      # bvec[row_offset] = max_load_diff
      bvec[row_offset] = self.avg_load + max_load_diff
      # load on server i >= avg_load - epsilon
      Amat.setrow(row_offset + 1, list(zip(nzidxs, nzvals2)))
      # bvec[row_offset + 1] = max_load_diff
      bvec[row_offset + 1] = -self.avg_load + max_load_diff
      row_offset += 2
      coo_data.extend(nzvals)
      coo_row.extend([row_offset - 2] * self.num_shards)
      coo_col.extend(nzidxs)
      coo_data.extend(nzvals2)
      coo_row.extend([row_offset - 1] * self.num_shards)
      coo_col.extend(nzidxs)

    ### Memory constraints [on x values] --> Inequality
    for i in range(self.num_servers):
      nzidxs = [self.num_vars + i * self.num_shards + j for j in range(self.num_shards)]
      nzvals = self.shard_sizes
      # memory on server i <= max_memory
      Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
      bvec[row_offset] = self.max_memory
      coo_data.extend(nzvals)
      coo_row.extend([row_offset] * self.num_shards)
      coo_col.extend(nzidxs)
      row_offset += 1
    ### Set r <= x  --> Inequality
    for i in range(self.num_servers):
      for j in range(self.num_shards):
        nzidxs = [i * self.num_shards + j, self.num_vars + i * self.num_shards + j]
        nzvals = [self.constraint_scale, -1 * self.constraint_scale]
        Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
        coo_data.extend(nzvals)
        coo_row.extend([row_offset] * 2)
        coo_col.extend(nzidxs)
        bvec[row_offset] = 0
        row_offset += 1
    ### Replication factor constraints --> Inequality
    for j in range(self.num_shards):
      nzidxs = [self.num_vars + i * self.num_shards + j for i in range(self.num_servers)]
      nzvals = [-1 * self.constraint_scale] * self.num_servers
      Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
      coo_data.extend(nzvals)
      coo_row.extend([row_offset] * self.num_servers)
      coo_col.extend(nzidxs)
      bvec[row_offset] = -1 * self.constraint_scale * self.replication_factor
      row_offset += 1
  
    if not self.only_inequalities:
      assert row_offset == mi, f"Row offset mismatch: {row_offset} != {mi}"
      ### Sum (r) = 1 --> Equality
      for j in range(self.num_shards):
        nzidxs = [i * self.num_shards + j for i in range(self.num_servers)]
        nzvals = [self.constraint_scale] * self.num_servers
        Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
        bvec[row_offset] = self.constraint_scale
        coo_data.extend(nzvals)
        coo_row.extend([row_offset] * self.num_servers)
        coo_col.extend(nzidxs)
        row_offset += 1
    else:
      ### Sum (r) = 1 --> 2xEquality
      for j in range(self.num_shards):
        nzidxs = [i * self.num_shards + j for i in range(self.num_servers)]
        nzvals = [self.constraint_scale] * self.num_servers
        nzvals2 = [-x for x in nzvals]
        Amat.setrow(row_offset, list(zip(nzidxs, nzvals)))
        Amat.setrow(row_offset + 1, list(zip(nzidxs, nzvals2)))
        coo_data.extend(nzvals)
        coo_row.extend([row_offset] * self.num_servers)
        coo_col.extend(nzidxs)
        coo_data.extend(nzvals2)
        coo_row.extend([row_offset + 1] * self.num_servers)
        coo_col.extend(nzidxs)
        bvec[row_offset] = self.constraint_scale
        bvec[row_offset + 1] = -self.constraint_scale
        row_offset += 2
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
    self.Acoo = coo_matrix((coo_data, (coo_row, coo_col)), shape=(mi + me, self.num_vars * 2))
    self.Acsr = self.Acoo.tocsr()

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
  solver = "ALCD"
  # Create ALCDWrapper object
  load_start_t = time.time()
  print(f"Creating ALCD format from inputs")
  only_inequalities = False
  lpobj = ALCDWrapper(shard_loads, shard_memory_usages, current_locations, num_servers, max_memory, only_inequalities)
  primal_args = lpobj.get_primal_alcd_format()
  dual_lpargs = lpobj.get_dual_alcd_format()
  load_end_t = time.time()

  if solver == "ALCD":
    # Create args for ALCD solver
    lpcfg = lps.LP_Param()
    lpcfg.solve_from_dual = False
    lpcfg.eta = 1
    lpcfg.verbose = True
    lpcfg.tol_trans = 1e-1
    lpcfg.tol = 1e-1
    # lpcfg.tol_sub = args.alcd_tol
    lpcfg.tol_sub = 1e-1
    lpcfg.use_CG = False
    lpcfg.max_iter = 100
    lpcfg.inner_max_iter = 20

    # Initialize ALCD solver
    A, b, c, nb, nf, m, me = primal_args
    print(f"Primal args: nb={nb}, nf={nf}, m={m}, me={me}")
    A.printrows()
    print(b)
    At = dual_lpargs[0]
    At.printrows()
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
    print(f"h2jj: {h2jj}\nhjj_ubound: {hjj_ubound}")
    print(f"h2jj: {list(zip(*np.histogram(h2jj, bins=10)))}")
    print(f"hjj_ubound: {list(zip(*np.histogram(hjj_ubound, bins=10)))}")
    # Solve via ALCD solver
    x0[:] = 1
    w0[:] = 1
      
    # Solve using ALCD
    lpinfo = lps.LP_Info()
    solve_start_time = time.time()
    print(f"Solving using ALCD solver")
    # lps.solve_alcd(A, b, c, x0, w0, h2jj, hjj_ubound, nb, nf, m, me, lpcfg, lpinfo)
    lps.solve_alcd(A, b, c, x0, w0, h2jj, hjj_ubound, nb, nf, m, me, lpcfg, lpinfo)
    solve_end_time = time.time()
  else:
    A, b, c, nb, nf, m, me = primal_args
    Acsr = lpobj.Acsr
    if not only_inequalities:
      # split Acsr into equalities and inequalities
      num_ineq, num_eq = m, me
      Aineq = Acsr[:num_ineq, :]
      Aeq = Acsr[num_ineq:, :]
      bineq = b[:num_ineq]
      beq = b[num_ineq:]
      print(f"Aineq: {Aineq}")
      print(f"bineq: {bineq}")
      print(f"Aeq: {Aeq}")
      print(f"beq: {beq}")
    else:
      print(f"Acsr: {Acsr}")
      print(f"b: {b}")
    print(f"c: {c}")
    # solve using cvxpy
    init_start_time = time.time()
    x0 = cp.Variable(len(c), nonneg=True)
    if not only_inequalities:
      # constraints = [Acsr @ x0 <= b]
      constraints = [Aineq @ x0 <= bineq, Aeq @ x0 == beq]
    else:
      constraints = [Acsr @ x0 <= b]
    obj = cp.Minimize(c @ x0)
    prob = cp.Problem(obj, constraints)
    init_end_time = time.time()
    solve_start_time = time.time()
    prob.solve(solver=cp.GLPK, verbose=True)
    solve_end_time = time.time()
    print(f"CVXPY Solver: Status={prob.status}, Optimal value={prob.value}")
    x0 = x0.value

  # Extract results
  rvars = x0[:num_shards*num_servers].round(3)
  xvars = x0[num_shards*num_servers:].round(3)
  # convert to list
  rvars = rvars.reshape((num_servers, num_shards)).tolist()
  xvars = xvars.reshape((num_servers, num_shards)).tolist()
  print(f"ALCD Solver: Load={(load_end_t - load_start_t)*1000:.1f}ms, Init: {(init_end_time - init_start_time)*1000:.1f}ms, Solve time: {(solve_end_time - solve_start_time)*1000:.1f}ms")
  return rvars, xvars