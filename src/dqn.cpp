
#include <cstdlib>
#include <vector>
#include <cmath>

#include <iostream>

#include "../lib/dqn.hpp"

void DQN::synchronize() {
    for(unsigned int l = 0; l < agent.num_of_layers(); l++) {
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++)
                target.layer(l)->node(n)->set_weight(i, agent.layer(l)->node(n)->weight(i));
        }
    }
}

void DQN::train(std::vector<std::vector<double>> &state, std::vector<std::vector<double>> &reward) {
    // learning parameters
    double EPSILON = 0.55;
    double GAMMA = 0.90;
    unsigned int BATCH_SIZE = 50;

    // replay memory
    unsigned int memory_size = 0;
    std::vector<unsigned int> action_memory;

    for(unsigned int t = 0; t < state.size(); t++) {
        unsigned int action_t;
        double explore = (double)rand() / RAND_MAX;
        // e-greedy policy
        if(explore < EPSILON)
            action_t = rand() % agent.layer(agent.num_of_layers() - 1)->out_features();
        else {
            std::vector<double> agent_q = agent.predict(state[t]);
            action_t = std::max_element(agent_q.begin(), agent_q.end()) - agent_q.begin();

            std::vector<double>().swap(agent_q);
        }

        action_memory.push_back(action_t);
        memory_size++;

        // learning
        if(memory_size >= BATCH_SIZE * 2) {
            std::vector<unsigned int> batch_index(memory_size, 0);
            std::iota(batch_index.begin(), batch_index.end(), 0);
            std::shuffle(batch_index.begin(), batch_index.end(), seed);

            for(unsigned int k = 0; k < BATCH_SIZE; k++) {
                double expected_reward, loss;
                double partial_gradient, gradient;
                unsigned int index = batch_index[k];

                // sample transition
                std::vector<double> agent_q = agent.predict(state[index]);
                if(index != memory_size - 1) {
                    std::vector<double> target_q = target.predict(state[index+1]);
                    expected_reward = reward[index][action_memory[index]] + GAMMA * (*std::max_element(target_q.begin(), target_q.end()));

                    std::vector<double>().swap(target_q);
                }
                else
                    expected_reward = reward[index][action_memory[index]];

                // SGD
                loss = expected_reward - agent_q[action_memory[index]];
            }

            std::vector<unsigned int>().swap(batch_index);
        }

    }
}

