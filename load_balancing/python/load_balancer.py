# Converted from LoadBalancer.java
# (suhasjs) --> verified
import cvxpy as cp
import numpy as np
from math import isclose
from alcd_load_balancer_lp_relaxed import *

class LoadBalancer:
    verbose = False
    min_replication_factor = 1
    epsilonRatio = 20
    solver = cp.GLPK

    def __init__(self):
        self.lastR = []
        self.lastX = []
        self.lastM = []
        self.lastNumServers = 0
        self.lastNumShards = 0

    def balance_load(self, num_shards, num_servers, shard_loads, shard_memory_usages, current_locations,
                     sample_queries, max_memory, split_factor=None, relax=False, alcd=False):
        if split_factor is not None:
            # Splitting logic
            return self._balance_load_with_splitting(num_shards, num_servers, shard_loads, 
                                                     shard_memory_usages, current_locations,
                                                     sample_queries, max_memory, split_factor)
        elif not relax:
            return self._balance_load_core(num_shards, num_servers, shard_loads, 
                                           shard_memory_usages, current_locations, 
                                           sample_queries, max_memory)
        elif not alcd:
            return self._balance_load_core_lp_relaxation(num_shards, num_servers, shard_loads, 
                                                         shard_memory_usages, current_locations, 
                                                         sample_queries, max_memory)
        else:
            return self._balance_load_core_lp_relaxation_alcd(num_shards, num_servers, shard_loads, 
                                                              shard_memory_usages, current_locations, 
                                                              sample_queries, max_memory)

    def _balance_load_with_splitting(self, num_shards, num_servers, shard_loads, shard_memory_usages, 
                                     current_locations, sample_queries, max_memory, split_factor):
        assert num_shards % split_factor == 0
        assert num_servers % split_factor == 0

        shard_loads_local = shard_loads[:]
        total_load = sum(shard_loads_local)
        servers_per_split = num_servers // split_factor
        load_per_split = round(total_load / split_factor)
        finalRs = [ [0.0]*num_shards for _ in range(num_servers) ]
        previous_splits_to_shards = {}

        for server_num in range(num_servers):
            split_num = server_num // servers_per_split
            if split_num not in previous_splits_to_shards:
                previous_splits_to_shards[split_num] = []
            for shard_num in range(num_shards):
                if current_locations[server_num][shard_num] == 1 and shard_num not in previous_splits_to_shards[split_num]:
                    previous_splits_to_shards[split_num].append(shard_num)

        new_splits_to_shards = {}
        new_splits_to_loads = {}
        splits_needing_more = {}

        # Fill splits from existing shards
        for split_num in range(split_factor):
            new_splits_to_shards[split_num] = []
            new_splits_to_loads[split_num] = []
            current_load = 0
            shard_list = sorted(previous_splits_to_shards.get(split_num, []), key=lambda x: shard_loads_local[x])
            for next_shard in shard_list:
                shard_load = shard_loads_local[next_shard]
                if shard_load == 0:
                    continue
                if current_load + shard_load <= load_per_split:
                    current_load += shard_load
                    new_splits_to_loads[split_num].append(shard_load)
                    shard_loads_local[next_shard] = 0
                    new_splits_to_shards[split_num].append(next_shard)
                    if current_load == load_per_split:
                        break
                else:
                    remaining = load_per_split - current_load
                    new_splits_to_loads[split_num].append(remaining)
                    current_load = load_per_split
                    shard_loads_local[next_shard] = shard_load - remaining
                    new_splits_to_shards[split_num].append(next_shard)
                    break
            if current_load < load_per_split:
                splits_needing_more[split_num] = current_load

        # Fill splits with leftover shards
        for split_num in splits_needing_more:
            current_load = splits_needing_more[split_num]
            for shard_num in range(num_shards):
                shard_load = shard_loads_local[shard_num]
                if shard_load == 0:
                    continue
                if current_load + shard_load <= load_per_split:
                    current_load += shard_load
                    new_splits_to_loads[split_num].append(shard_load)
                    shard_loads_local[shard_num] = 0
                    new_splits_to_shards[split_num].append(shard_num)
                    if current_load == load_per_split:
                        break
                else:
                    remaining = load_per_split - current_load
                    new_splits_to_loads[split_num].append(remaining)
                    current_load = load_per_split
                    shard_loads_local[shard_num] = shard_load - remaining
                    new_splits_to_shards[split_num].append(shard_num)
                    break
            # Simple check for rounding errors

        # Now solve each split independently as if it’s a smaller problem
        for split_num in range(split_factor):
            split_shards = new_splits_to_shards[split_num]
            split_shard_loads = new_splits_to_loads[split_num]
            num_split_shards = len(split_shards)
            local_shard_loads = [0]*num_split_shards
            local_shard_memory_usages = [0]*num_split_shards

            for i in range(num_split_shards):
                shard_id = split_shards[i]
                local_shard_loads[i] = split_shard_loads[i]
                local_shard_memory_usages[i] = shard_memory_usages[shard_id]

            local_current_locations = []
            for server_idx in range(servers_per_split):
                global_server_num = split_num * servers_per_split + server_idx
                row = []
                for i in range(num_split_shards):
                    shard_id = split_shards[i]
                    row.append(current_locations[global_server_num][shard_id])
                local_current_locations.append(row)

            # sampleQueries is empty assert:
            rs = self._balance_load_core(num_split_shards, servers_per_split, local_shard_loads,
                                         local_shard_memory_usages, local_current_locations, 
                                         sample_queries, max_memory)

            # Copy solution into final
            for server_idx in range(servers_per_split):
                global_server_num = split_num * servers_per_split + server_idx
                for i in range(num_split_shards):
                    finalRs[global_server_num][split_shards[i]] = rs[server_idx][i]

        return finalRs

    def _balance_load_core(self, num_shards, num_servers, shard_loads, shard_memory_usages, current_locations, 
                           sample_queries, max_memory):
        # Parallel objective
        solver1_m = None
        sample_query_keys = [k for k in sample_queries.keys() if len(k) > 1]
        sample_query_keys = sorted(sample_query_keys, key=lambda k: sample_queries[k], reverse=True)
        max_query_samples = 500
        sample_query_keys = sample_query_keys[:max_query_samples]
        num_sample_queries = len(sample_query_keys)

        # Just do integer variables with CVXPY style placeholders
        r_vars = []
        x_vars = []
        for _ in range(num_servers):
            r_vars.append([cp.Variable(nonneg=True) for __ in range(num_shards)])
            x_vars.append([cp.Variable(boolean=True) for __ in range(num_shards)])

        # We approximate the "m" array
        m_vars = [cp.Variable(boolean=True) for _ in range(num_sample_queries)]
        query_weights = [sample_queries[k] for k in sample_query_keys]

        constraints = []
        # Link parallel obj constraints
        for server_num in range(num_servers):
            for q_idx, shard_set in enumerate(sample_query_keys):
                # sum x for shards in shard_set <= m[q_idx]
                expr = cp.sum([x_vars[server_num][s] for s in shard_set])
                constraints.append(expr <= m_vars[q_idx])

        # Objective: minimize sum(m * queryWeights)
        parallel_obj = cp.Minimize(sum([m_vars[i] * query_weights[i] for i in range(num_sample_queries)]))

        # Core constraints
        constraints += self._set_core_constraints(r_vars, x_vars, num_shards, num_servers,
                                                  shard_loads, shard_memory_usages, max_memory)

        # Solve
        prob1 = cp.Problem(parallel_obj, constraints)
        if num_sample_queries > 0:
            prob1.solve(solver=LoadBalancer.solver, verbose=LoadBalancer.verbose)
            solver1_m = [var.value for var in m_vars]
        else:
            solver1_m = [20]*len(m_vars)

        # Transfer objective
        # Redefine r_vars, x_vars
        r_vars2 = []
        x_vars2 = []
        for _ in range(num_servers):
            r_vars2.append([cp.Variable(nonneg=True) for __ in range(num_shards)])
            x_vars2.append([cp.Variable(boolean=True) for __ in range(num_shards)])

        # Transfer cost
        transfer_costs = np.copy(current_locations)
        # compute cost as 1 if not present, 0 otherwise
        for i in range(len(transfer_costs)):
            for j in range(len(transfer_costs[i])):
                transfer_costs[i][j] = 1 if transfer_costs[i][j] == 0 else 0

        transfer_cost_list = [sum([x_vars2[i][j] * transfer_costs[i][j] for j in range(num_shards)]) for i in range(num_servers)]
        transfer_obj = cp.Minimize(cp.sum(transfer_cost_list))

        constraints2 = []
        # link constraints to m
        q_idx = 0
        for shard_set in sample_query_keys:
            for server_num in range(num_servers):
                expr = sum([x_vars2[server_num][s] for s in shard_set])
                constraints2.append(expr <= solver1_m[q_idx])
            q_idx += 1

        constraints2 += self._set_core_constraints(r_vars2, x_vars2, num_shards, num_servers, 
                                                   shard_loads, shard_memory_usages, max_memory)

        prob2 = cp.Problem(transfer_obj, constraints2)
        prob2.solve(solver=LoadBalancer.solver, verbose=LoadBalancer.verbose)

        # Retrieve final results
        self.lastNumShards = num_shards
        self.lastNumServers = num_servers
        self.lastR = []
        self.lastX = []

        for i in range(num_servers):
            row_r = [r_vars2[i][j].value for j in range(num_shards)]
            row_x = [x_vars2[i][j].value for j in range(num_shards)]
            self.lastR.append(row_r)
            self.lastX.append(row_x)

        return self.lastR

    def _balance_load_core_lp_relaxation_alcd(self, num_shards, num_servers, shard_loads, shard_memory_usages,
                                            current_locations, sample_queries, max_memory):
        rs, xs = balance_load_alcd(num_shards, num_servers, shard_loads, shard_memory_usages, 
                                   current_locations, sample_queries, max_memory)
        
        # Retrieve final results
        self.lastNumShards = num_shards
        self.lastNumServers = num_servers
        self.lastR = []
        self.lastX = []

        for i in range(num_servers):
            row_r = [rs[i][j] for j in range(num_shards)]
            row_x = [xs[i][j] for j in range(num_shards)]
            self.lastR.append(row_r)
            self.lastX.append(row_x)

        # compute binary-ness of the solution in x
        def test_binaryness(vars, printstr):
            np_xvars = np.array(vars)
            binary_vals = np.isclose(np_xvars, 0, rtol=1e-3) | np.isclose(np_xvars, 1, rtol=1e-3)
            nonbinary_vals = np.logical_not(binary_vals)
            num_binary_vals = np.sum(binary_vals)
            num_nonbinary_vals = np.sum(nonbinary_vals)
            perc_non_binary = num_nonbinary_vals / (num_servers*num_shards) * 100
            print(f"{printstr} --> Total: {num_servers * num_shards}, # binary: {num_binary_vals}, # non-binary: {num_nonbinary_vals} ({perc_non_binary:.2f}%), histogram: {np.histogram(np_xvars, bins=10)}")
        '''
        # round X up to 1 if r > 0
        for i in range(num_servers):
            for j in range(num_shards):
                if self.lastR[i][j] > 0:
                    self.lastX[i][j] = 1
                else:
                    self.lastX[i][j] = 0
        '''
        # check for memory violations
        for i in range(num_servers):
            server_memory_usage = sum([self.lastX[i][j] * shard_memory_usages[j] for j in range(num_shards)])
            if server_memory_usage > max_memory:
                print(f"Memory violation for server {i}, usage: {server_memory_usage} > {max_memory}")
        test_binaryness(self.lastR, "R")
        test_binaryness(self.lastX, "X")
        newR = self._fix_memory_violations(self.lastR, shard_loads, shard_memory_usages, max_memory)
        self.lastR = newR
        test_binaryness(newR, "R")
        # recompute new x
        for i in range(num_servers):
            for j in range(num_shards):
                self.lastX[i][j] = 1 if newR[i][j] > 0 else 0
        test_binaryness(self.lastX, "X")
        return self.lastR
    
    def _balance_load_core_lp_relaxation(self, num_shards, num_servers, shard_loads, shard_memory_usages,
                                         current_locations, sample_queries, max_memory):
        # Parallel objective
        solver1_m = None
        sample_query_keys = [k for k in sample_queries.keys() if len(k) > 1]
        sample_query_keys = sorted(sample_query_keys, key=lambda k: sample_queries[k], reverse=True)
        max_query_samples = 500
        sample_query_keys = sample_query_keys[:max_query_samples]
        num_sample_queries = len(sample_query_keys)

        # LP relaxation for x_vars
        r_vars = []
        x_vars = []
        for _ in range(num_servers):
            r_vars.append([cp.Variable(nonneg=True) for __ in range(num_shards)])
            x_vars.append([cp.Variable(nonneg=True) for __ in range(num_shards)])

        # We approximate the "m" array
        m_vars = [cp.Variable(boolean=True) for _ in range(num_sample_queries)]
        query_weights = [sample_queries[k] for k in sample_query_keys]

        constraints = []
        # Link parallel obj constraints
        for server_num in range(num_servers):
            for q_idx, shard_set in enumerate(sample_query_keys):
                # sum x for shards in shard_set <= m[q_idx]
                expr = cp.sum([x_vars[server_num][s] for s in shard_set])
                constraints.append(expr <= m_vars[q_idx])

        # Objective: minimize sum(m * queryWeights)
        parallel_obj = cp.Minimize(sum([m_vars[i] * query_weights[i] for i in range(num_sample_queries)]))

        # Core constraints
        constraints += self._set_core_constraints(r_vars, x_vars, num_shards, num_servers,
                                                shard_loads, shard_memory_usages, max_memory)

        # Solve
        if num_sample_queries > 0:
            prob1 = cp.Problem(parallel_obj, constraints)
            prob1.solve(solver=LoadBalancer.solver, verbose=LoadBalancer.verbose)
            solver1_m = [var.value for var in m_vars]
        else:
            solver1_m = [20]*len(m_vars)

        # Transfer objective
        # Redefine r_vars, x_vars
        r_vars2 = []
        x_vars2 = []
        for _ in range(num_servers):
            r_vars2.append([cp.Variable(nonneg=True) for __ in range(num_shards)])
            x_vars2.append([cp.Variable(nonneg=True) for __ in range(num_shards)])

        # Transfer cost
        transfer_costs = np.copy(current_locations)
        # compute cost as 1 if not present, 0 otherwise
        for i in range(len(transfer_costs)):
            for j in range(len(transfer_costs[i])):
                transfer_costs[i][j] = 1 if transfer_costs[i][j] == 0 else 0

        transfer_cost_list = [sum([x_vars2[i][j] * transfer_costs[i][j] for j in range(num_shards)]) for i in range(num_servers)]
        norm_lambda = 1e3
        xvars_arr = []
        for i in range(num_servers):
            xvars_arr += x_vars2[i]
        # transfer_obj = cp.Minimize(cp.sum(transfer_cost_list) + norm_lambda * cp.norm(cp.vstack(xvars_arr), p=2))
        transfer_obj = cp.Minimize(cp.sum(transfer_cost_list))

        constraints2 = []
        # link constraints to m
        q_idx = 0
        for shard_set in sample_query_keys:
            for server_num in range(num_servers):
                expr = sum([x_vars2[server_num][s] for s in shard_set])
                constraints2.append(expr <= solver1_m[q_idx])
            q_idx += 1

        constraints2 += self._set_core_constraints(r_vars2, x_vars2, num_shards, num_servers, 
                                                   shard_loads, shard_memory_usages, max_memory)

        prob2 = cp.Problem(transfer_obj, constraints2)
        # print(f"Problem: {prob2}")
        # prob2.solve(solver=LoadBalancer.solver, verbose=LoadBalancer.verbose, **{'scipy_options': {'method': 'highs-ipm', 'disp': True, 'tol': 1e-5}})
        prob2.solve(solver=LoadBalancer.solver, verbose=LoadBalancer.verbose)
        print(f"[LP relaxation] Solver status: {prob2.status}, objective: {prob2.value}")

        # Retrieve final results
        self.lastNumShards = num_shards
        self.lastNumServers = num_servers
        self.lastR = []
        self.lastX = []

        for i in range(num_servers):
            row_r = [r_vars2[i][j].value for j in range(num_shards)]
            row_x = [x_vars2[i][j].value for j in range(num_shards)]
            self.lastR.append(row_r)
            self.lastX.append(row_x)

        # compute binary-ness of the solution in x
        def test_binaryness(vars, printstr):
            np_xvars = np.array(vars)
            binary_vals = np.isclose(np_xvars, 0, rtol=1e-3) | np.isclose(np_xvars, 1, rtol=1e-3)
            nonbinary_vals = np.logical_not(binary_vals)
            num_binary_vals = np.sum(binary_vals)
            num_nonbinary_vals = np.sum(nonbinary_vals)
            perc_non_binary = num_nonbinary_vals / (num_servers*num_shards) * 100
            print(f"{printstr} --> Total: {num_servers * num_shards}, # binary: {num_binary_vals}, # non-binary: {num_nonbinary_vals} ({perc_non_binary:.2f}%), histogram: {np.histogram(np_xvars, bins=10)}")
        # round X up to 1 if r > 0
        for i in range(num_servers):
            for j in range(num_shards):
                if self.lastR[i][j] > 0:
                    self.lastX[i][j] = 1
        # check for memory violations
        for i in range(num_servers):
            server_memory_usage = sum([self.lastX[i][j] * shard_memory_usages[j] for j in range(num_shards)])
            if server_memory_usage > max_memory:
                print(f"Memory violation for server {i}, usage: {server_memory_usage} > {max_memory}")
        test_binaryness(self.lastR, "R")
        test_binaryness(self.lastX, "X")
        newR = self._fix_memory_violations(self.lastR, shard_loads, shard_memory_usages, max_memory)
        self.lastR = newR
        test_binaryness(newR, "R")
        # recompute new x
        for i in range(num_servers):
            for j in range(num_shards):
                self.lastX[i][j] = 1 if newR[i][j] > 0 else 0
        test_binaryness(self.lastX, "X")
        return self.lastR

    def _set_core_constraints(self, r_vars, x_vars, num_shards, num_servers,
                              shard_loads, shard_memory_usages, max_memory):
        constraints = []
        actual_rep_factor = self.min_replication_factor if self.min_replication_factor < num_servers else num_servers
        avg_load = sum(shard_loads) / num_servers
        epsilonRatio = 20.0
        epsilon = avg_load / epsilonRatio

        # Load constraints
        for i in range(num_servers):
            expr_load = cp.sum([r_vars[i][j]*shard_loads[j] for j in range(num_shards)])
            constraints.append(expr_load <= avg_load + epsilon)
            constraints.append(expr_load >= avg_load - epsilon)

        # Memory constraints
        for i in range(num_servers):
            expr_mem = cp.sum([x_vars[i][j]*shard_memory_usages[j] for j in range(num_shards)])
            constraints.append(expr_mem <= max_memory)

        # Link r <= x, also optional constraint if replicationFactor>1
        for i in range(num_servers):
            for j in range(num_shards):
                constraints.append(r_vars[i][j] <= x_vars[i][j])
                if actual_rep_factor > 1:
                    # x(i,j) <= r(i,j) + 0.9999
                    constraints.append(x_vars[i][j] <= r_vars[i][j] + 0.9999)

        # Sum of r for each shard = 1
        for j in range(num_shards):
            constraints.append(cp.sum([r_vars[i][j] for i in range(num_servers)]) == 1.0)

        # Replication factor constraint
        for j in range(num_shards):
            constraints.append(cp.sum([x_vars[i][j] for i in range(num_servers)]) >= actual_rep_factor)

        return constraints

    def _fix_memory_violations(self, rvars, shard_loads, shard_memory_usages, max_memory):
        num_servers = len(rvars)
        num_shards = len(rvars[0])
        # construct x vars, compute memory usage
        xvars = []
        memory_usages = [0] * num_servers
        for i in range(num_servers):
            xvars.append([1 if rvars[i][j] > 0 else 0 for j in range(num_shards)])
            memory_usages[i] = sum([xvars[i][j] * shard_memory_usages[j] for j in range(num_shards)])
        violated_servers = [i for i in range(num_servers) if memory_usages[i] > max_memory]
        violated_servers = sorted(violated_servers, key=lambda i: memory_usages[i], reverse=True)
        # fix memory violations
        for server_id in violated_servers:
            print(f"Memory violation: {server_id}, allocated: {memory_usages[server_id]} > {max_memory}")
            # sort shards assigned to the server by load in increasing order of load
            sorted_shards = sorted(list(range(num_shards)), key=lambda j: rvars[server_id][j]*shard_loads[j])
            for shard_id in sorted_shards:
                # check if memory violation is fixed
                if memory_usages[server_id] <= max_memory:
                    break
                # is this shard replicated ? If not then skip
                num_replicas = sum([xvars[i][shard_id] for i in range(num_servers)])
                if num_replicas == 1:
                    continue
                # remove shard from server
                removed_load = rvars[server_id][shard_id]
                rvars[server_id][shard_id] = 0
                xvars[server_id][shard_id] = 0
                # redistribute this load onto other replicas
                for i in range(num_servers):
                    if xvars[i][shard_id] > 0:
                        # proportionally redistribute the load
                        added_load = (rvars[i][shard_id] / (1 - removed_load)) * removed_load
                        rvars[i][shard_id] += added_load
                memory_usages[server_id] -= shard_memory_usages[shard_id]
        return rvars

    @staticmethod
    def heuristic_balance(shard_loads, shard_map, servers_list):
        epsilonRatio = 20.0
        lost_shards = set()
        gained_shards = set()
        server_loads = {sv: 0 for sv in servers_list}
        server_to_shards = {sv: [] for sv in servers_list}
        for sh, ld in shard_loads.items():
            server_num = shard_map[sh]
            server_loads[server_num] += ld
            server_to_shards[server_num].append(sh)

        sorted_by_min = list(servers_list)
        sorted_by_max = list(servers_list)
        # Priority queue approach in python, replicate with simple sorts
        average_load = sum(shard_loads.values()) / len(servers_list)
        epsilon = average_load / epsilonRatio
        return_map = dict(shard_map)

        def sort_queues():
            sorted_by_min.sort(key=lambda s: server_loads[s])
            sorted_by_max.sort(key=lambda s: server_loads[s], reverse=True)

        sort_queues()
        while len(sorted_by_max) > 0 and server_loads[sorted_by_max[0]] > average_load + epsilon:
            # pick the most loaded server, remove it from the queue
            over_loaded_server = sorted_by_max[0]
            sorted_by_max = sorted_by_max[1:]
            while (len(server_to_shards[over_loaded_server]) > 0 and 
                   server_loads[over_loaded_server] > average_load + epsilon):
                # pick the least loaded server, remove it from the queue
                under_loaded_server = sorted_by_min[0]
                sorted_by_min = sorted_by_min[1:]
                # pick the largest load shard
                shards_over = [sh for sh in server_to_shards[over_loaded_server] if shard_loads[sh] > 0]
                if not shards_over:
                    break
                most_loaded_shard = max(shards_over, key=lambda x: shard_loads[x])
                server_to_shards[over_loaded_server].remove(most_loaded_shard)
                # can move most loaded shard to least loaded server
                if server_loads[under_loaded_server] + shard_loads[most_loaded_shard] <= average_load + epsilon:
                    return_map[most_loaded_shard] = under_loaded_server
                    server_loads[over_loaded_server] -= shard_loads[most_loaded_shard]
                    server_loads[under_loaded_server] += shard_loads[most_loaded_shard]
                    lost_shards.add(over_loaded_server)
                    gained_shards.add(under_loaded_server)
                sorted_by_min.append(under_loaded_server)
                sort_queues()
            sort_queues()
        return return_map