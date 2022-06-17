#ifndef __DQN_HPP_
#define __DQN_HPP_

#include <vector>

#include "neural_network.hpp"

class DQN
{
private:
    NeuralNetwork agent;
    NeuralNetwork target;
public:
    DQN() {}
    DQN(std::vector<std::vector<unsigned int>> shape);
};

#endif
