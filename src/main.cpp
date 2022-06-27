
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include "../lib/dqn.hpp"

std::vector<std::vector<double>> read(std::string path) {
    std::ifstream file(path);
    std::vector<std::vector<double>> dat;

    if(file.is_open()) {
        std::string line;
        while(std::getline(file, line)) {
            std::vector<double> row;
            unsigned int start = 0;
            for(unsigned int i = 0; i < line.length(); i++) {
                if(line[i] == ' ' || i == line.length() - 1) {
                    row.push_back(std::stod(line.substr(start, i - start)));
                    start = i + 1;
                }
            }
            dat.push_back(row);
            std::vector<double>().swap(row);
        }

        file.close();
    }

    return dat;
}

int main(int argc, char *argv[])
{
    std::vector<std::vector<double>> state = read("./data/residual");
    std::vector<std::vector<double>> reward = read("./data/reward");

    DQN dqn({{30,18},{18,12},{12,6},{6,3}});

    unsigned int ITERATION = 10;
    unsigned int BATCH_SIZE = 512;
    double ALPHA = 0.001;
    double ALPHA_DECAY = 0.999;
    double EPSILON = 0.90;
    double EPSILON_DECAY = 0.999;
    double GAMMA = 0.99;
    unsigned int SYNC_FREQUENCY = 1000;

    dqn.optimize(state, reward, ITERATION, BATCH_SIZE, ALPHA, ALPHA_DECAY, EPSILON, EPSILON_DECAY, GAMMA, SYNC_FREQUENCY);

    return 0;
}
