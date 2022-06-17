
#include <cstdlib>
#include <vector>
#include <random>

#include "../lib/dqn.hpp"

DQN::DQN(std::vector<std::vector<unsigned int>> shape) {
    for(unsigned int l = 0; l < shape.size(); l++) {
        agent.add_layer(shape[l][0], shape[l][1]);
        target.add_layer(shape[l][0], shape[l][1]);
    }

    std::default_random_engine seed;
    agent.initialize(seed);
    target.initialize(seed);
}
