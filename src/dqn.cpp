
#include <cstdlib>
#include <vector>
#include <cmath>

#include <iostream>

#include "../lib/bar.hpp"
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
    double EPSILON = 0.05;
    double GAMMA = 0.90;
    double ALPHA = 0.001;
    unsigned int SYNCHRONIZATION = 10;

    unsigned int DATASET_SIZE = state.size();
    unsigned int BATCH_SIZE = 100;
    unsigned int ITERATION = 100;

    unsigned int memory_size = 0;
    std::vector<unsigned int> action_memory;

    for(unsigned int t = 0; t < state.size(); t++) {
        // e-greedy policy
        unsigned int action_t;
        double explore = (double)rand() / RAND_MAX;
        if(explore < EPSILON)
            action_t = rand() % agent.layer(agent.num_of_layers() - 1)->out_features();
        else {
            std::vector<double> agent_q = agent.predict(state[t]);
            action_t = std::max_element(agent_q.begin(), agent_q.end()) - agent_q.begin();

            std::vector<double>().swap(agent_q);
        }

        action_memory.push_back(action_t);
        memory_size++;

        // mini-batch learning
        if(memory_size >= BATCH_SIZE * 2) {
            std::vector<unsigned int> batch_index(memory_size, 0);
            std::iota(batch_index.begin(), batch_index.end(), 0);
            std::shuffle(batch_index.begin(), batch_index.end(), seed);
            batch_index.erase(batch_index.begin() + BATCH_SIZE, batch_index.end());

            for(unsigned int itr = 1; itr <= ITERATION; itr++) {
                std::vector<double> expected_reward;
                for(unsigned int k = 0; k < BATCH_SIZE; k++) {
                    unsigned int index = batch_index[k];
                    // compute expected reward (finite bellman equation)
                    double y = reward[index][action_memory[index]];
                    if(index != memory_size - 1) {
                        std::vector<double> target_q = target.predict(state[index+1]);
                        y += GAMMA * (*std::max_element(target_q.begin(), target_q.end()));

                        std::vector<double>().swap(target_q);
                    }
                    expected_reward.push_back(y);

                    // SGD
                    std::vector<double> agent_q = agent.predict(state[index]);
                    for(int l = agent.num_of_layers() - 1; l >= 0; l--) {
                        unsigned int start = 0;
                        unsigned int end = agent.layer(l)->out_features();
                        if(l == agent.num_of_layers() - 1) {
                             start = action_memory[index];
                             end = start + 1;
                        }

                        double partial_gradient = 0.00, gradient = 0.00;
                        for(unsigned int n = start; n < end; n++) {
                            if(l == agent.num_of_layers() - 1)
                                partial_gradient = -2.00 * (y - agent_q[action_memory[index]]);
                            else {
                                partial_gradient = agent.layer(l)->node(n)->err() * relu_prime(agent.layer(l)->node(n)->sum());

                                double updated_bias = agent.layer(l)->node(n)->bias() - ALPHA * partial_gradient;
                                agent.layer(l)->node(n)->set_bias(updated_bias);
                            }

                            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                                if(l > 0) {
                                    gradient = partial_gradient * agent.layer(l-1)->node(i)->act();
                                    agent.layer(l-1)->node(i)->add_err(partial_gradient * agent.layer(l)->node(n)->weight(i));
                                }
                                else
                                    gradient = partial_gradient * state[index][i];

                                gradient += 1.00 / DATASET_SIZE * agent.layer(l)->node(n)->weight(i); // L2 regularization

                                double updated_weight = agent.layer(l)->node(n)->weight(i) - ALPHA * gradient;
                                agent.layer(l)->node(n)->set_weight(i, updated_weight);
                            }
                        }
                    }
                    std::vector<double>().swap(agent_q);
                }

                if(itr % SYNCHRONIZATION == 0)
                    synchronize();

                double loss = 0.00;
                for(unsigned int k = 0; k < BATCH_SIZE; k++) {
                    unsigned int index = batch_index[k];
                    std::vector<double> agent_q = agent.predict(state[index]);
                    loss += pow(expected_reward[k] - agent_q[action_memory[index]], 2);

                    std::vector<double>().swap(agent_q);
                }
                loss /= BATCH_SIZE;

                std::vector<double>().swap(expected_reward);

                progress_bar(itr, ITERATION, "LOSS = " + std::to_string(loss));
            }
        }
    }
}

