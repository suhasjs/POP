# Converted from LoadBalancerExecutable.java
# (suhasjs) --> verified
from load_balancer import LoadBalancer

import argparse
import time
import random
import copy
import numpy as np
import pickle

class LBRunner:
    num_shards = 0
    num_servers = 0
    split_factor = 4
    max_memory = 16
    num_rounds = 100
    skip_rounds = 20
    random_seed = 42
    scale_factor = 1e5
    load_type = "stateless"
    percent_change = 30
    zipf_constant = 0.25
    zipf_scale = 0.5
    constant_change = True
    logfile = None
    logs = []

    @staticmethod
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--numShards", type=int, required=True, help="Number of shards")
        parser.add_argument("--numServers", type=int, required=True, help="Number of servers")
        parser.add_argument("--numSplits", type=int, default=1, help="Split factor for POP")
        parser.add_argument("--numRounds", type=int, default=5, help="Number of rounds to run")
        parser.add_argument("--randomSeed", type=int, default=0, help="Random seed")
        parser.add_argument("--load", type=str, help="Type of load? [stateless, stateful]", choices=["stateless", "stateful"], default="stateless")
        parser.add_argument("--percentChange", type=int, default=LBRunner.percent_change, help="How much does zipf value change between two rounds (only for stateful load generator)")
        parser.add_argument("--benchmark", type=str, required=True, help="Which benchmark to run ", choices=["base", "base-lp-relaxed", "base-lp-relaxed-alcd", "split", "heuristic"])
        parser.add_argument("--logfile", type=str, required=False, help="Output run log to a pkl file ", default=None)
        args = parser.parse_args()

        LBRunner.num_shards = args.numShards
        LBRunner.num_servers = args.numServers
        LBRunner.split_factor = args.numSplits
        LBRunner.num_rounds = args.numRounds
        LBRunner.random_seed = args.randomSeed
        LBRunner.load_type = args.load
        LBRunner.percent_change = args.percentChange
        LBRunner.logfile = args.logfile
        benchmark = args.benchmark

        if benchmark == "base":
            LBRunner.zipfian_benchmark()
        elif benchmark == "base-lp-relaxed":
            LBRunner.zipfian_lprelaxed_benchmark()
        elif benchmark == "base-lp-relaxed-alcd":
            LBRunner.zipfian_lprelaxed_alcd_benchmark()
        elif benchmark == "split":
            LBRunner.zipfian_benchmark_split()
        elif benchmark == "heuristic":
            LBRunner.zipfian_heuristic_benchmark()
        
        if LBRunner.logfile is not None:
            print(f"Saving logs to {LBRunner.logfile}")
            with open(LBRunner.logfile, "wb") as f:
                pickle.dump(LBRunner.logs, f)
    
    @staticmethod
    def load_generator(previous_loads, load_scale_factor, other_params):
        random_gen = other_params['random_gen']
        new_loads = [0] * LBRunner.num_shards
        if LBRunner.load_type == "stateless" or len(previous_loads) == 0:
            zipf_value = 0.25 + random_gen.random() * 0.5
            for shard_num in range(LBRunner.num_shards):
                load_val = int(round(load_scale_factor * (1.0 / ((shard_num+1)**zipf_value))))
                new_loads[shard_num] = load_val
        else:
            previous_zipf_value = other_params['prev_zipf_value']
            perc_change = LBRunner.percent_change
            random_val = random_gen.random()
            if LBRunner.constant_change:
                change_val = (perc_change / 100) * previous_zipf_value
                current_change = change_val if random_val > 0.5 else -change_val
            else:
                current_change = ((random_val - 0.5) * perc_change / 100) * previous_zipf_value
            zipf_value = previous_zipf_value + current_change
            print(f"Zipf value: {previous_zipf_value:.3f} + {current_change:.3f} --> {zipf_value:.3f}")
            for shard_num in range(LBRunner.num_shards):
                load_val = int(round(load_scale_factor * (1.0 / ((shard_num+1)**zipf_value))))
                new_loads[shard_num] = load_val
        return new_loads, zipf_value
    @staticmethod
    def process_load_distribution(return_r, shard_loads, current_locations):
        new_locations = copy.deepcopy(current_locations)
        server_loads = []
        for server_num, Rs in enumerate(return_r):
            server_load = 0
            for i, val in enumerate(Rs):
                server_load += val * shard_loads[i]
                new_locations[server_num][i] = 1 if val > 0 else 0
            server_loads.append(server_load)
        
        # compute load imbalance, shards moved
        server_loads.sort()
        avg_load = sum(shard_loads) / float(LBRunner.num_servers)
        shards_moved = 0
        load_imbalance = ((server_loads[-1] - server_loads[0]) / avg_load) * 100.0
        for i in range(LBRunner.num_servers):
            for j in range(LBRunner.num_shards):
                if new_locations[i][j] == 1 and current_locations[i][j] == 0:
                    # track movement
                    shards_moved += 1
        num_replicas = [0]*LBRunner.num_shards
        for i in range(LBRunner.num_shards):
            for j in range(LBRunner.num_servers):
                if current_locations[j][i] == 1:
                    num_replicas[i] += 1
        return new_locations, load_imbalance, shards_moved, num_replicas

    @staticmethod
    def zipfian_benchmark():
        print(f"Running base benchmark with {LBRunner.num_shards} shards x {LBRunner.num_servers} servers")
        lb = LoadBalancer()
        lb.verbose = False
        current_locations = []
        for server_num in range(LBRunner.num_servers):
            row = [0]*LBRunner.num_shards
            current_locations.append(row)
        for shard_num in range(LBRunner.num_shards):
            server_num = shard_num % LBRunner.num_servers
            current_locations[server_num][shard_num] = 1

        total_time = 0
        total_movements = 0
        total_imbalance = 0
        r = random.Random()
        r.seed(LBRunner.random_seed)
        previous_loads = []
        prev_zipf_value = None
        for round_num in range(LBRunner.num_rounds):
            shard_loads, zipf_value = LBRunner.load_generator(previous_loads, LBRunner.scale_factor, {"random_gen": r, "prev_zipf_value": prev_zipf_value})
            previous_loads, prev_zipf_value = shard_loads, zipf_value
            # shard_loads = [0]*LBRunner.num_shards
            memory_usages = [1]*LBRunner.num_shards
            # for shard_num in range(LBRunner.num_shards):
            #     load_val = int(round(scale_factor * (1.0 / ((shard_num+1)**zipf_value))))
            #     shard_loads[shard_num] = load_val
            average_load = sum(shard_loads) / float(LBRunner.num_servers)
            print(f"Average load: {average_load:.2f}, target load range: [{average_load*0.9:.2f}, {average_load*1.1:.2f}]")

            start_time = time.time()
            return_r = lb.balance_load(LBRunner.num_shards, LBRunner.num_servers,
                                       shard_loads, memory_usages, current_locations, {}, LBRunner.max_memory)
            lb_time = (time.time() - start_time) * 1000.0
            assert len(return_r) == LBRunner.num_servers

            last_locations = current_locations
            new_stats = LBRunner.process_load_distribution(return_r, shard_loads, last_locations)
            current_locations, load_imbalance, shards_moved, num_replicas = new_stats
            sorted_num_replicas = sorted(num_replicas, reverse=True)
            print(f"Top # replicas: {sorted_num_replicas[:10]}")
            # shards_moved = 0
            # server_loads = []
            # for server_num, Rs in enumerate(return_r):
            #     server_load = 0
            #     for i, val in enumerate(Rs):
            #         server_load += val * shard_loads[i]
            #         current_locations[server_num][i] = 1 if val > 0 else 0
            #     server_loads.append(server_load)

            #     # Quick checks
            #     # assert server_load <= average_load * 1.1, f"Server load: {server_load}, Average load: {average_load} --> expected server load <= average load * 1.1"
            #     # assert server_load >= average_load * 0.9, f"Server load: {server_load}, Average load: {average_load} --> expected server load >= average load * 0.9"
            # # compute load imbalance
            # server_loads.sort()
            # load_imbalance = ((server_loads[-1] / server_loads[0]) - 1) * 100.0
            # for i in range(LBRunner.num_servers):
            #     for j in range(LBRunner.num_shards):
            #         if current_locations[i][j] == 1 and last_locations[i][j] == 0:
            #             # track movement
            #             shards_moved += 1

            if round_num > LBRunner.skip_rounds:
                total_movements += shards_moved
                total_time += lb_time
                total_imbalance += load_imbalance
            step_log = {
                "round" : round_num, "zipf" : zipf_value, "shards_moved" : shards_moved,
                "load_imbalance" : load_imbalance, "lb_time_ms" : lb_time,
                "num_shards" : LBRunner.num_shards, "num_servers" : LBRunner.num_servers,
                "shard_loads" : shard_loads, "new_x" : current_locations, "new_r" : return_r,
                "num_replicas" : num_replicas
                }
            LBRunner.logs.append(step_log)

            print(f"Round: {round_num} Zipf: {zipf_value:.3f} Shards Moved: {shards_moved} Imbalance: {load_imbalance:.2f}% LB time: {int(lb_time)}ms")

        avg_moves = float(total_movements) / (LBRunner.num_rounds - LBRunner.skip_rounds)
        avg_time = int(total_time / (LBRunner.num_rounds - LBRunner.skip_rounds))
        avg_imbalance = float(total_imbalance) / (LBRunner.num_rounds - LBRunner.skip_rounds)
        print(f"Average movements: {avg_moves:.2f}, Average time: {avg_time}ms, Average imbalance: {avg_imbalance:.2f}%")

    @staticmethod
    def zipfian_lprelaxed_benchmark():
        print(f"Running base [LP-relaxation] benchmark with {LBRunner.num_shards} shards x {LBRunner.num_servers} servers")
        lb = LoadBalancer()
        lb.verbose = False
        current_locations = []
        for server_num in range(LBRunner.num_servers):
            row = [0]*LBRunner.num_shards
            current_locations.append(row)
        for shard_num in range(LBRunner.num_shards):
            server_num = shard_num % LBRunner.num_servers
            current_locations[server_num][shard_num] = 1

        total_time = 0
        total_movements = 0
        total_imbalance = 0
        r = random.Random()
        r.seed(LBRunner.random_seed)
        previous_loads = []
        prev_zipf_value = None
        for round_num in range(LBRunner.num_rounds):
            shard_loads, zipf_value = LBRunner.load_generator(previous_loads, LBRunner.scale_factor, {"random_gen": r, "prev_zipf_value": prev_zipf_value})
            previous_loads, prev_zipf_value = shard_loads, zipf_value
            # shard_loads = [0]*LBRunner.num_shards
            memory_usages = [1]*LBRunner.num_shards
            # for shard_num in range(LBRunner.num_shards):
            #     load_val = int(round(scale_factor * (1.0 / ((shard_num+1)**zipf_value))))
            #     shard_loads[shard_num] = load_val
            average_load = sum(shard_loads) / float(LBRunner.num_servers)
            print(f"Average load: {average_load:.2f}, target load range: [{average_load*0.9:.2f}, {average_load*1.1:.2f}]")

            start_time = time.time()
            return_r = lb.balance_load(LBRunner.num_shards, LBRunner.num_servers,
                                       shard_loads, memory_usages, current_locations, {}, LBRunner.max_memory, relax=True)
            lb_time = (time.time() - start_time) * 1000.0
            assert len(return_r) == LBRunner.num_servers

            last_locations = current_locations
            new_stats = LBRunner.process_load_distribution(return_r, shard_loads, last_locations)
            current_locations, load_imbalance, shards_moved, num_replicas = new_stats
            sorted_num_replicas = sorted(num_replicas, reverse=True)
            print(f"Top # replicas: {sorted_num_replicas[:10]}")
            # last_locations = copy.deepcopy(current_locations)
            # shards_moved = 0
            # server_loads = []
            # for server_num, Rs in enumerate(return_r):
            #     server_load = 0
            #     for i, val in enumerate(Rs):
            #         server_load += val * shard_loads[i]
            #         current_locations[server_num][i] = 1 if val > 0 else 0
            #     server_loads.append(server_load)

            #     # Quick checks
            #     # assert server_load <= average_load * 1.1, f"Server load: {server_load}, Average load: {average_load} --> expected server load <= average load * 1.1"
            #     # assert server_load >= average_load * 0.9, f"Server load: {server_load}, Average load: {average_load} --> expected server load >= average load * 0.9"
            # # compute load imbalance
            # server_loads.sort()
            # load_imbalance = ((server_loads[-1] / server_loads[0]) - 1) * 100.0
            # for i in range(LBRunner.num_servers):
            #     for j in range(LBRunner.num_shards):
            #         if current_locations[i][j] == 1 and last_locations[i][j] == 0:
            #             # track movement
            #             shards_moved += 1

            if round_num > LBRunner.skip_rounds:
                total_movements += shards_moved
                total_time += lb_time
                total_imbalance += load_imbalance
            step_log = {
                "round" : round_num, "zipf" : zipf_value, "shards_moved" : shards_moved,
                "load_imbalance" : load_imbalance, "lb_time_ms" : lb_time,
                "num_shards" : LBRunner.num_shards, "num_servers" : LBRunner.num_servers,
                "shard_loads" : shard_loads, "new_x" : current_locations, "new_r" : return_r,
                "num_replicas" : num_replicas
                }
            LBRunner.logs.append(step_log)

            print(f"Round: {round_num} Zipf: {zipf_value:.3f} Shards Moved: {shards_moved} Imbalance: {load_imbalance:.2f}% LB time: {int(lb_time)}ms")

        avg_moves = float(total_movements) / (LBRunner.num_rounds - LBRunner.skip_rounds)
        avg_time = int(total_time / (LBRunner.num_rounds - LBRunner.skip_rounds))
        avg_imbalance = float(total_imbalance) / (LBRunner.num_rounds - LBRunner.skip_rounds)
        print(f"Average movements: {avg_moves:.2f}, Average time: {avg_time}ms, Average imbalance: {avg_imbalance:.2f}%")

    @staticmethod
    def zipfian_lprelaxed_alcd_benchmark():
        print(f"Running base [LP-relaxation] [ALCD] benchmark with {LBRunner.num_shards} shards x {LBRunner.num_servers} servers")
        lb = LoadBalancer()
        lb.verbose = False
        current_locations = []
        for server_num in range(LBRunner.num_servers):
            row = [0]*LBRunner.num_shards
            current_locations.append(row)
        for shard_num in range(LBRunner.num_shards):
            server_num = shard_num % LBRunner.num_servers
            current_locations[server_num][shard_num] = 1

        total_time = 0
        total_movements = 0
        total_imbalance = 0
        r = random.Random()
        r.seed(LBRunner.random_seed)
        previous_loads = []
        prev_zipf_value = None
        for round_num in range(LBRunner.num_rounds):
            shard_loads, zipf_value = LBRunner.load_generator(previous_loads, LBRunner.scale_factor, {"random_gen": r, "prev_zipf_value": prev_zipf_value})
            previous_loads, prev_zipf_value = shard_loads, zipf_value
            # shard_loads = [0]*LBRunner.num_shards
            memory_usages = [1]*LBRunner.num_shards
            # for shard_num in range(LBRunner.num_shards):
            #     load_val = int(round(scale_factor * (1.0 / ((shard_num+1)**zipf_value))))
            #     shard_loads[shard_num] = load_val
            average_load = sum(shard_loads) / float(LBRunner.num_servers)
            print(f"Average load: {average_load:.2f}, target load range: [{average_load*0.9:.2f}, {average_load*1.1:.2f}]")

            start_time = time.time()
            return_r = lb.balance_load(LBRunner.num_shards, LBRunner.num_servers,
                                       shard_loads, memory_usages, current_locations, {}, LBRunner.max_memory, relax=True, alcd=True)
            lb_time = (time.time() - start_time) * 1000.0
            assert len(return_r) == LBRunner.num_servers

            last_locations = current_locations
            new_stats = LBRunner.process_load_distribution(return_r, shard_loads, last_locations)
            current_locations, load_imbalance, shards_moved, num_replicas = new_stats
            sorted_num_replicas = sorted(num_replicas, reverse=True)
            print(f"Top #replicas: {sorted_num_replicas[:10]}")
            # last_locations = copy.deepcopy(current_locations)
            # shards_moved = 0
            # server_loads = []
            # for server_num, Rs in enumerate(return_r):
            #     server_load = 0
            #     for i, val in enumerate(Rs):
            #         server_load += val * shard_loads[i]
            #         current_locations[server_num][i] = 1 if val > 0 else 0
            #     server_loads.append(server_load)

            #     # Quick checks
            #     # assert server_load <= average_load * 1.1, f"Server load: {server_load}, Average load: {average_load} --> expected server load <= average load * 1.1"
            #     # assert server_load >= average_load * 0.9, f"Server load: {server_load}, Average load: {average_load} --> expected server load >= average load * 0.9"
            # # compute load imbalance
            # server_loads.sort()
            # load_imbalance = ((server_loads[-1] / server_loads[0]) - 1) * 100.0
            # for i in range(LBRunner.num_servers):
            #     for j in range(LBRunner.num_shards):
            #         if current_locations[i][j] == 1 and last_locations[i][j] == 0:
            #             # track movement
            #             shards_moved += 1

            if round_num > LBRunner.skip_rounds:
                total_movements += shards_moved
                total_time += lb_time
                total_imbalance += load_imbalance
            step_log = {
                "round" : round_num, "zipf" : zipf_value, "shards_moved" : shards_moved,
                "load_imbalance" : load_imbalance, "lb_time_ms" : lb_time,
                "num_shards" : LBRunner.num_shards, "num_servers" : LBRunner.num_servers,
                "shard_loads" : shard_loads, "new_x" : current_locations, "new_r" : return_r,
                "num_replicas" : num_replicas
                }
            LBRunner.logs.append(step_log)

            print(f"Round: {round_num} Zipf: {zipf_value:.3f} Shards Moved: {shards_moved} Imbalance: {load_imbalance:.2f}% LB time: {int(lb_time)}ms")

        avg_moves = float(total_movements) / (LBRunner.num_rounds - LBRunner.skip_rounds)
        avg_time = int(total_time / (LBRunner.num_rounds - LBRunner.skip_rounds))
        avg_imbalance = float(total_imbalance) / (LBRunner.num_rounds - LBRunner.skip_rounds)
        print(f"Average movements: {avg_moves:.2f}, Average time: {avg_time}ms, Average imbalance: {avg_imbalance:.2f}%")

    @staticmethod
    def zipfian_benchmark_split():
        lb = LoadBalancer()
        lb.verbose = False
        current_locations = [[0]*LBRunner.num_shards for _ in range(LBRunner.num_servers)]
        total_time = 0
        total_movements = 0
        total_imbalance = 0
        order = list(range(LBRunner.num_shards))
        random.shuffle(order)
        r = random.Random()
        r.seed(LBRunner.random_seed)
        previous_loads = []
        prev_zipf_value = None

        for round_num in range(LBRunner.num_rounds):
            shard_loads, zipf_value = LBRunner.load_generator(previous_loads, LBRunner.scale_factor, {"random_gen": r, "prev_zipf_value": prev_zipf_value})
            previous_loads, prev_zipf_value = shard_loads, zipf_value
            # shard_loads = [0]*LBRunner.num_shards
            memory_usages = [1]*LBRunner.num_shards
            # for shard_num in range(LBRunner.num_shards):
            #     load_val = int(round(scale_factor * (1.0 / ((shard_num+1)**zipf_value))))
            #     shard_loads[shard_num] = load_val

            start_time = time.time()
            return_r = lb.balance_load(LBRunner.num_shards, LBRunner.num_servers,
                                       shard_loads, memory_usages, current_locations, {}, 
                                       LBRunner.max_memory, LBRunner.split_factor)
            lb_time = (time.time() - start_time) * 1000.0
            assert len(return_r) == LBRunner.num_servers
        
            last_locations = current_locations
            new_stats = LBRunner.process_load_distribution(return_r, shard_loads, last_locations)
            current_locations, load_imbalance, shards_moved, num_replicas = new_stats
            # last_locations = copy.deepcopy(current_locations)
            # shards_moved = 0
            # for server_num, Rs in enumerate(return_r):
            #     for i, val in enumerate(Rs):
            #         current_locations[server_num][i] = 1 if val > 0 else 0
            # for i in range(LBRunner.num_servers):
            #     for j in range(LBRunner.num_shards):
            #         if current_locations[i][j] == 1 and last_locations[i][j] == 0:
            #             shards_moved += 1

            if round_num >= LBRunner.skip_rounds:
                total_movements += shards_moved
                total_time += lb_time
                total_imbalance += load_imbalance
            step_log = {
                "round" : round_num, "zipf" : zipf_value, "shards_moved" : shards_moved,
                "load_imbalance" : load_imbalance, "lb_time_ms" : lb_time,
                "num_shards" : LBRunner.num_shards, "num_servers" : LBRunner.num_servers,
                "shard_loads" : shard_loads, "new_x" : current_locations, "new_r" : return_r,
                "num_replicas" : num_replicas
                }
            LBRunner.logs.append(step_log)
            print(f"Round: {round_num} Zipf: {zipf_value:.3f} Shards Moved: {shards_moved} Imbalance: {load_imbalance:.2f}% LB time: {int(lb_time)}ms")

        avg_moves = float(total_movements) / (LBRunner.num_rounds - LBRunner.skip_rounds)
        avg_time = int(total_time / (LBRunner.num_rounds - LBRunner.skip_rounds))
        avg_imbalance = float(total_imbalance) / (LBRunner.num_rounds - LBRunner.skip_rounds)
        print(f"Split Average movements: {avg_moves:.2f}, Average time: {avg_time}ms, Average imbalance: {avg_imbalance:.2f}%")

    @staticmethod
    def zipfian_heuristic_benchmark():
        lb = LoadBalancer()
        lb.verbose = False
        current_locations = {}
        for i in range(LBRunner.num_shards):
            current_locations[i] = i % LBRunner.num_servers
        servers_list = list(range(LBRunner.num_servers))

        total_time = 0
        total_movements = 0
        total_imbalance = 0
        r = random.Random()
        r.seed(LBRunner.random_seed)
        previous_loads = []
        prev_zipf_value = None
        for round_num in range(LBRunner.num_rounds):
            gen_shard_loads, zipf_value = LBRunner.load_generator(previous_loads, LBRunner.scale_factor, {"random_gen": r, "prev_zipf_value": prev_zipf_value})
            previous_loads, prev_zipf_value = gen_shard_loads, zipf_value
            total_load = sum(gen_shard_loads)
            # shard_loads = [0]*LBRunner.num_shards
            memory_usages = [1]*LBRunner.num_shards
            # for shard_num in range(LBRunner.num_shards):
            #     load_val = int(round(scale_factor * (1.0 / ((shard_num+1)**zipf_value))))
            #     shard_loads[shard_num] = load_val

            zipf_value = 0.25 + r.random() * 0.5
            shard_loads = {i : gen_shard_loads[i] for i in range(LBRunner.num_shards)}

            last_locations = copy.deepcopy(current_locations)
            start_time = time.time()
            current_locations = LoadBalancer.heuristic_balance(shard_loads, current_locations, servers_list)
            lb_time = (time.time() - start_time)*1000.0
            assert len(current_locations) == LBRunner.num_shards
            average_load = total_load / float(LBRunner.num_servers)
            server_loads = [0]*LBRunner.num_servers

            shards_moved = 0
            for sh in range(LBRunner.num_shards):
                server_loads[current_locations[sh]] += shard_loads[sh]
                if current_locations[sh] != last_locations[sh]:
                    shards_moved += 1
            server_loads.sort()
            load_imbalance = ((server_loads[-1] - server_loads[0]) / average_load) * 100.0

            if round_num >= LBRunner.skip_rounds:
                total_movements += shards_moved
                total_time += lb_time
                total_imbalance += load_imbalance
            step_log = {
                "round" : round_num, "zipf" : zipf_value, "shards_moved" : shards_moved,
                "load_imbalance" : load_imbalance, "lb_time_ms" : lb_time,
                "num_shards" : LBRunner.num_shards, "num_servers" : LBRunner.num_servers,
                "shard_loads" : shard_loads, "new_x" : current_locations, "new_r" : None,
                "num_replicas" : [1]*LBRunner.num_shards
                }
            LBRunner.logs.append(step_log)

            print(f"Round: {round_num} Zipf: {zipf_value:.3f} Shards Moved: {shards_moved} LB time: {int(lb_time)}ms, Imbalance: {load_imbalance:.2f}%")

        avg_moves = float(total_movements) / (LBRunner.num_rounds - LBRunner.skip_rounds)
        avg_time = int(total_time / (LBRunner.num_rounds - LBRunner.skip_rounds))
        avg_imbalance = float(total_imbalance) / (LBRunner.num_rounds - LBRunner.skip_rounds)
        print(f"Average movements: {avg_moves:.2f}, Average time: {avg_time}ms, Average imbalance: {avg_imbalance:.2f}%")

if __name__ == "__main__":
    LBRunner.main()