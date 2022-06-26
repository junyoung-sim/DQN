#ifndef __DQN_HPP_
#define __DQN_HPP_

#include <vector>
#include <random>
#include <chrono>
#include <string>

#include "neural_network.hpp"

class DQN
{
private:
    NeuralNetwork agent;
    NeuralNetwork target;
    std::default_random_engine seed;
public:
    DQN() {}
    DQN(std::vector<std::vector<unsigned int>> shape) {
        for(unsigned int l = 0; l < shape.size(); l++) {
            unsigned int in = shape[l][0], out = shape[l][1];
            agent.add_layer(in, out);
            target.add_layer(in, out);
        }

        seed.seed(std::chrono::system_clock::now().time_since_epoch().count());
        agent.initialize(seed);
        sync();
    }
    ~DQN() {}

    void sync();

    std::vector<double> agent_performance(std::vector<std::vector<double>> &state, std::vector<std::vector<double>> &reward, double GAMMA);

    void optimize(std::vector<std::vector<double>> &state, std::vector<std::vector<double>> &reward,
                  unsigned int ITERATION, unsigned int BATCH_SIZE, double ALPHA, double ALPHA_DECAY, double EPSILON, double EPSILON_DECAY, double GAMMA, unsigned int SYNC_FREQUENCY);
};

#endif
