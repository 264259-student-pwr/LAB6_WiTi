#include <mpi.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_set>

// Task structure
struct Task {
    int id;
    double p;  // processing time
    double w;  // weight
    double d;  // deadline
};

// Read tasks from file
std::vector<Task> read_tasks(const std::string& filename) {
    std::ifstream infile(filename);
    std::string line;
    int n;
    infile >> n;
    std::vector<Task> tasks(n);
    for (int i = 0; i < n; ++i) {
        infile >> tasks[i].id >> tasks[i].p >> tasks[i].w >> tasks[i].d;
    }
    return tasks;
}

// Calculate tardiness
double calculate_tardiness(const std::vector<int>& order, const std::vector<Task>& tasks) {
    double C = 0;
    double total_cost = 0;
    for (int i : order) {
        C += tasks[i].p;
        double T = std::max(0.0, C - tasks[i].d);
        total_cost += tasks[i].w * T;
    }
    return total_cost;
}

// Genetic Algorithm functions
std::vector<int> initialize_permutation(int n) {
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(perm.begin(), perm.end(), g);
    return perm;
}

std::vector<std::vector<int>> initialize_population(int n, int pop_size) {
    std::vector<std::vector<int>> population;
    for (int i = 0; i < pop_size; ++i) {
        population.push_back(initialize_permutation(n));
    }
    return population;
}

std::pair<int, int> select_parents(const std::vector<double>& probabilities) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(probabilities.begin(), probabilities.end());

    int parent1 = dis(gen);
    int parent2 = dis(gen);
    while (parent1 == parent2) {
        parent2 = dis(gen);
    }
    return { parent1, parent2 };
}

std::vector<int> crossover(const std::vector<int>& parent1, const std::vector<int>& parent2) {
    int size = parent1.size();
    std::vector<int> child(size, -1);
    std::unordered_set<int> taken;

    int crossover_point = std::rand() % size;
    for (int i = 0; i < crossover_point; ++i) {
        child[i] = parent1[i];
        taken.insert(parent1[i]);
    }

    int fill_pos = crossover_point;
    for (int i = 0; i < size; ++i) {
        if (taken.find(parent2[i]) == taken.end()) {
            child[fill_pos++] = parent2[i];
            taken.insert(parent2[i]);
        }
    }
    return child;
}

void mutate(std::vector<int>& perm, double mutation_rate) {
    int size = perm.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < size; ++i) {
        if (dis(gen) < mutation_rate) {
            int j = std::rand() % size;
            std::swap(perm[i], perm[j]);
        }
    }
}

// MPI-based fitness evaluation
std::vector<double> evaluate_population(const std::vector<std::vector<int>>& population, const std::vector<Task>& tasks, int rank, int size) {
    int pop_size = population.size();
    int tasks_per_proc = pop_size / size;
    int start = rank * tasks_per_proc;
    int end = (rank == size - 1) ? pop_size : start + tasks_per_proc;

    std::vector<double> local_fitnesses(end - start);
    for (int i = start; i < end; ++i) {
        local_fitnesses[i - start] = calculate_tardiness(population[i], tasks);
    }

    std::vector<double> global_fitnesses(pop_size);
    MPI_Allgather(local_fitnesses.data(), end - start, MPI_DOUBLE, global_fitnesses.data(), end - start, MPI_DOUBLE, MPI_COMM_WORLD);
    return global_fitnesses;
}

// Main GA loop
std::pair<std::vector<int>, double> genetic_algorithm(const std::vector<Task>& tasks, int pop_size = 100, int num_generations = 100, double mutation_rate = 0.1) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = tasks.size();
    std::vector<std::vector<int>> population = initialize_population(n, pop_size);

    for (int generation = 0; generation < num_generations; ++generation) {
        std::vector<double> fitnesses = evaluate_population(population, tasks, rank, size);

        // Convert fitness to probabilities
        double fitness_sum = std::accumulate(fitnesses.begin(), fitnesses.end(), 0.0);
        std::vector<double> probabilities(fitnesses.size());
        for (size_t i = 0; i < fitnesses.size(); ++i) {
            probabilities[i] = fitnesses[i] / fitness_sum;
        }

        std::vector<std::vector<int>> new_population;
        for (int i = 0; i < pop_size / 2; ++i) {
            auto parents = select_parents(probabilities);
            auto child1 = crossover(population[parents.first], population[parents.second]);
            auto child2 = crossover(population[parents.second], population[parents.first]);
            mutate(child1, mutation_rate);
            mutate(child2, mutation_rate);
            new_population.push_back(child1);
            new_population.push_back(child2);
        }
        population = new_population;
    }

    std::vector<double> final_fitnesses = evaluate_population(population, tasks, rank, size);
    auto best_it = std::min_element(final_fitnesses.begin(), final_fitnesses.end());
    int best_idx = std::distance(final_fitnesses.begin(), best_it);

    return { population[best_idx], *best_it };
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string filename = argv[1];
    auto tasks = read_tasks(filename);

    auto result = genetic_algorithm(tasks);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        const auto& best_order = result.first;
        double best_fitness = result.second;

        std::cout << "Best order: ";
        for (int i : best_order) {
            std::cout << tasks[i].id << " ";
        }
        std::cout << "\nBest fitness: " << best_fitness << std::endl;
    }

    MPI_Finalize();
    return 0;
}