package edu.stanford.futuredata;

import ilog.concert.IloException;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LoadBalancerExecutable {

    private static final Logger logger = LoggerFactory.getLogger(LoadBalancerExecutable.class);

    static int numShards;
    static int numServers;
    static int splitFactor = 4;
    static int maxMemory = 16;
    static int numRounds = 100;
    static int skipRounds = 20;
    static long randomSeed = 42;
    // stateful: random 10% change in zipf value each round
    static double[] statefulZipfValues = new double[]{0.672,0.739,0.665,0.599,0.659,0.593,0.652,0.587,0.528,0.581,0.639,0.703,0.633,0.696,0.750,0.675,0.743,0.750,0.750,0.750,0.675,0.743,0.750,0.750,0.675,0.608,0.547,0.601,0.662,0.728,0.655,0.720,0.648,0.713,0.750,0.675,0.743,0.668,0.735,0.750,0.675,0.608,0.668,0.601,0.541,0.595,0.536,0.589,0.531,0.584,0.642,0.578,0.520,0.468,0.515,0.566,0.510,0.561,0.617,0.678,0.746,0.750,0.750,0.750,0.750,0.675,0.743,0.668,0.735,0.662,0.595,0.536,0.589,0.648,0.584,0.525,0.578,0.635,0.699,0.750,0.750,0.750,0.750,0.675,0.743,0.668,0.735,0.750,0.750,0.750,0.750,0.750,0.675,0.743,0.750,0.750,0.750,0.675,0.743,0.668};
    // random: 0.25 + random_value * 0.5 where random_value is random between 0 and 1
    static double[] randomZipfValues = new double[]{0.614,0.592,0.404,0.389,0.583,0.702,0.434,0.388,0.482,0.641,0.710,0.468,0.625,0.443,0.339,0.547,0.355,0.663,0.336,0.544,0.626,0.536,0.540,0.626,0.266,0.429,0.659,0.459,0.737,0.607,0.490,0.396,0.725,0.660,0.568,0.435,0.430,0.467,0.479,0.486,0.485,0.664,0.326,0.667,0.480,0.390,0.348,0.340,0.683,0.493,0.460,0.566,0.600,0.408,0.536,0.436,0.686,0.653,0.558,0.437,0.599,0.704,0.348,0.655,0.564,0.482,0.403,0.520,0.568,0.313,0.499,0.264,0.268,0.492,0.578,0.448,0.720,0.442,0.573,0.635,0.697,0.549,0.738,0.289,0.638,0.389,0.700,0.437,0.466,0.416,0.408,0.471,0.298,0.623,0.335,0.653,0.320,0.296,0.264,0.524};


    public static void main(String[] args) throws Exception {
        Options options = new Options();
        options.addOption("numShards", true, "Number of Shards?");
        options.addOption("numServers", true, "Number of Servers?");
        options.addOption("numSplits", true, "Split Factor?");
        options.addOption("benchmark", true, "Which Benchmark?");
        options.addOption("stateful", true, "Whether to use stateful zipf values");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        numShards = Integer.parseInt(cmd.getOptionValue("numShards"));
        numServers = Integer.parseInt(cmd.getOptionValue("numServers"));
        splitFactor = cmd.hasOption("numSplits") ? Integer.parseInt(cmd.getOptionValue("numSplits")) : 1;
        String benchmark = cmd.getOptionValue("benchmark");
        boolean useStateful = cmd.hasOption("stateful");

        // double[] randomZipfValues = new double[numRounds];
        // Random r = new Random();
        // r.setSeed(randomSeed);
        // for (int roundNum = 0; roundNum < numRounds; roundNum++) {
        //     double zipfValue = 0.25 + r.nextDouble() * 0.5;
        //     randomZipfValues[roundNum] = zipfValue;
        // }
        double[] chosenZipfValues = useStateful ? statefulZipfValues : randomZipfValues;

        if (benchmark.equals("base")) {
            zipfianBenchmark(chosenZipfValues);
        } else if (benchmark.equals("split")) {
            zipfianBenchmarkSplit(chosenZipfValues);
        } else if (benchmark.equals("heuristic")) {
            zipfianHeuristicBenchmark(chosenZipfValues);
        }

    }

    public static void zipfianBenchmark(double[] zipfValues) throws IloException {
        logger.info("zipfianBenchmark");

        LoadBalancer.verbose = true;
        int[][] currentLocations = new int[numServers][numShards];
        for (int shardNum = 0; shardNum < numShards; shardNum++) {
            int serverNum = shardNum % numServers;
            currentLocations[serverNum][shardNum] = 1;
        }
        long totalTime = 0;
        int totalMovements = 0;
        Random r = new Random();
        r.setSeed(randomSeed);
        for (int roundNum = 0; roundNum < numRounds; roundNum++) {
            // double zipfValue = 0.25 + r.nextDouble() * 0.5;
            double zipfValue = zipfValues[roundNum];
            int[] shardLoads = new int[numShards];
            int[] memoryUsages = new int[numShards];
            for (int shardNum = 0; shardNum < numShards; shardNum++) {
                int load = (int) Math.round(1000000.0 * (1.0 / Math.pow(shardNum + 1, zipfValue)));
                shardLoads[shardNum] = load;
                memoryUsages[shardNum] = 1;
            }

            long startTime = System.currentTimeMillis();
            List<double[]> returnR = new LoadBalancer().balanceLoad(numShards, numServers, shardLoads, memoryUsages, currentLocations, new HashMap<>(), maxMemory);
            long lbTime = System.currentTimeMillis() - startTime;
            assertEquals(numServers, returnR.size());
            double averageLoad = IntStream.of(shardLoads).sum() / (double) numServers;
            int[][] lastLocations = new int[numServers][];
            for (int i = 0; i < numServers; i++) {
                lastLocations[i] = currentLocations[i].clone();
            }
            for (int serverNum = 0; serverNum < numServers; serverNum ++) {
                double[] Rs = returnR.get(serverNum);
                double serverLoad = 0;
                for (int i = 0; i < numShards; i++) {
                    serverLoad += Rs[i] * shardLoads[i];
                    currentLocations[serverNum][i] = Rs[i] > 0 ? 1 : 0;
                }
                assertTrue(serverLoad <= averageLoad * 1.1);
                assertTrue(serverLoad >= averageLoad * 0.9);
            }
            int shardsMoved = 0;
            for(int i = 0; i < numServers; i++) {
                for(int j = 0; j < numShards; j++) {
                    if (currentLocations[i][j] == 1 && lastLocations[i][j] == 0) {
                        shardsMoved++;
                    }
                }
            }
            if (roundNum >= skipRounds) {
                totalMovements += shardsMoved;
                totalTime += lbTime;
            }
            System.out.printf("Round: %d Zipf: %.3f Shards Moved: %d LB time: %dms\n", roundNum, zipfValue, shardsMoved, lbTime);
        }
        System.out.printf("Average movements: %.2f, Average time: %dms\n", (double) totalMovements / (numRounds - skipRounds), totalTime / (numRounds - skipRounds));
    }

    public static void zipfianBenchmarkSplit(double[] zipfValues) throws IloException {
        logger.info("zipfianBenchmarkSplit");
        LoadBalancer.verbose = false;
        int[][] currentLocations = new int[numServers][numShards];
        long totalTime = 0;
        int totalMovements = 0;
        List<Integer> order = IntStream.range(0, numShards).boxed().collect(Collectors.toList());
        Collections.shuffle(order);
        Random r = new Random();
        r.setSeed(randomSeed);
        for (int roundNum = 0; roundNum < numRounds; roundNum++) {
            // double zipfValue = 0.25 + r.nextDouble() * 0.5;
            double zipfValue = zipfValues[roundNum];
            int[] shardLoads = new int[numShards];
            int[] memoryUsages = new int[numShards];
            for (int shardNum = 0; shardNum < numShards; shardNum++) {
                int load = (int) Math.round(1000000.0 * (1.0 / Math.pow(shardNum + 1, zipfValue)));
                shardLoads[order.get(shardNum)] = load;
                memoryUsages[shardNum] = 1;
            }

            long startTime = System.currentTimeMillis();
            List<double[]> returnR = new LoadBalancer().balanceLoad(numShards, numServers, shardLoads, memoryUsages, currentLocations, new HashMap<>(), maxMemory, splitFactor);
            long lbTime = System.currentTimeMillis() - startTime;
            assertEquals(numServers, returnR.size());
            int[][] lastLocations = new int[numServers][];
            for (int i = 0; i < numServers; i++) {
                lastLocations[i] = currentLocations[i].clone();
            }
            for (int serverNum = 0; serverNum < numServers; serverNum ++) {
                double[] Rs = returnR.get(serverNum);
                for (int i = 0; i < numShards; i++) {
                    currentLocations[serverNum][i] = Rs[i] > 0 ? 1 : 0;
                }
            }
            int shardsMoved = 0;
            for(int i = 0; i < numServers; i++) {
                for(int j = 0; j < numShards; j++) {
                    if (currentLocations[i][j] == 1 && lastLocations[i][j] == 0) {
                        shardsMoved++;
                    }
                }
            }
            if (roundNum >= skipRounds) {
                totalMovements += shardsMoved;
                totalTime += lbTime;
            }
            System.out.printf("Round: %d Zipf: %.3f Shards Moved: %d LB time: %dms\n", roundNum, zipfValue, shardsMoved, lbTime);
        }
        System.out.printf("Split Average movements: %.2f, Average time: %dms\n", (double) totalMovements / (numRounds - skipRounds), totalTime / (numRounds - skipRounds));
    }

    public static void zipfianHeuristicBenchmark(double[] zipfValues) {
        logger.info("zipfianHeuristicBenchmark");

        LoadBalancer.verbose = false;
        Map<Integer, Integer> currentLocations = new HashMap<>();
        for (int i = 0; i < numShards; i++) {
            currentLocations.put(i, i % numServers);
        }
        List<Integer> serversList = new ArrayList<>();
        for (int i = 0 ; i < numServers; i++) {
            serversList.add(i);
        }
        long totalTime = 0;
        int totalMovements = 0;
        Random r = new Random();
        r.setSeed(randomSeed);
        for (int roundNum = 0; roundNum < numRounds; roundNum++) {
            // double zipfValue = 0.25 + r.nextDouble() * 0.5;
            double zipfValue = zipfValues[roundNum];
            Map<Integer, Integer> shardLoads = new HashMap<>();
            int totalLoad = 0;
            for (int shardNum = 0; shardNum < numShards; shardNum++) {
                int load = (int) Math.round(1000000.0 * (1.0 / Math.pow(shardNum + 1, zipfValue)));
                shardLoads.put(shardNum, load);
                totalLoad += load;
            }

            Map<Integer, Integer> lastLocations = new HashMap<>(currentLocations);
            long startTime = System.currentTimeMillis();
            currentLocations = LoadBalancer.heuristicBalance(shardLoads, currentLocations, serversList);
            long lbTime = System.currentTimeMillis() - startTime;
            assertEquals(numShards, currentLocations.size());
            double averageLoad = totalLoad / (double) numServers;

            // Check correctness.
            if (LoadBalancer.verbose) {
                for (int serverNum = 0; serverNum < numServers; serverNum++) {
                    double serverLoad = 0;
                    for (int shardNum = 0; shardNum < numShards; shardNum++) {
                        if (currentLocations.get(shardNum) == serverNum) {
                            serverLoad += shardLoads.get(shardNum);
                        }
                    }
                    logger.info("{} {} {}", serverNum, averageLoad, serverLoad);
                }
            }

            int shardsMoved = 0;
            for(int shardNum = 0; shardNum < numShards; shardNum++) {
                if (!currentLocations.get(shardNum).equals(lastLocations.get(shardNum))) {
                    shardsMoved++;
                }
            }
            if (roundNum >= skipRounds) {
                totalMovements += shardsMoved;
                totalTime += lbTime;
            }
            System.out.printf("Round: %d Zipf: %.3f Shards Moved: %d LB time: %dms\n", roundNum, zipfValue, shardsMoved, lbTime);
        }
        System.out.printf("Average movements: %.2f, Average time: %dms\n", (double) totalMovements / (numRounds - skipRounds), totalTime / (numRounds - skipRounds));
    }
}
