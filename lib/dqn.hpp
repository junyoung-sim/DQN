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
            agent.add_layer(shape[l][0], shape[l][1]);
            target.add_layer(shape[l][0], shape[l][1]);
        }

        seed.seed(std::chrono::system_clock::now().time_since_epoch().count());
        agent.initialize(seed);
        synchronize();
    }

    void synchronize();

    void train(std::vector<std::vector<double>> &state, std::vector<std::vector<double>> &reward);
};

#endif
