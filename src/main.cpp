
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>

#include "../lib/dqn.hpp"

int main(int argc, char *argv[])
{
/*
    NeuralNetwork model;

    model.add_layer(5, 10);
    model.add_layer(10, 10);
    model.add_layer(10, 3);

    std::default_random_engine seed;
    seed.seed(std::chrono::system_clock::now().time_since_epoch().count());
    model.initialize(seed);

    std::vector<double> test_x = {0.1, 0.3, -0.5, -0.9, 0.2};
    std::vector<double> yhat = model.predict(test_x);

    for(double &val: yhat)
        std::cout << val << " ";
*/

    std::vector<std::vector<double>> state, reward;

    // read state and reward from ./data/residual and ./data/reward

    DQN dqn({{15,10},{10,5},{5,3}});

    return 0;
}
