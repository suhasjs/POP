# Converted from LoadBalancerTests.java
# (suhasjs) --> verified
import unittest
from load_balancer import LoadBalancer

class LoadBalancerTests(unittest.TestCase):

    def test_balance_load_function(self):
        num_shards = 4
        num_servers = 2
        shard_loads = [1, 2, 3, 20]
        memory_usages = [9, 1, 1, 1]
        current_locations = [
            [1, 1, 1, 1],
            [0, 0, 0, 0]
        ]
        max_memory = 10
        float_eps = 1e-2

        lb = LoadBalancer()
        return_r = lb.balance_load(num_shards, num_servers, shard_loads, memory_usages, current_locations, {}, max_memory)
        average_load = sum(shard_loads) / float(num_servers)
        print(f"Return R: {return_r}")
        for Rs in return_r:
            server_load = 0.0
            for i, loadval in enumerate(shard_loads):
                server_load += Rs[i] * loadval
            print(f"Server load: {server_load}, Average load: {average_load}, load_range: {average_load * 0.95} - {average_load * 1.05}")
            self.assertTrue(server_load >= average_load * 0.95 - float_eps)
            self.assertTrue(server_load <= average_load * 1.05 + float_eps)

    def test_balance_load_heuristic(self):
        shard_loads = {0: 5, 1: 5, 2: 5, 3: 15}
        shard_map = {0: 0, 1: 0, 2: 0, 3: 0}
        servers_list = [0, 1]

        result = LoadBalancer.heuristic_balance(shard_loads, shard_map, servers_list)
        self.assertNotEqual(result[0], result[3])
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[0], result[2])

if __name__ == '__main__':
    unittest.main()