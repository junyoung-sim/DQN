
#include <cstdlib>
#include <vector>
#include <cmath>
#include <fstream>

#include "../lib/bar.hpp"
#include "../lib/dqn.hpp"

void DQN::sync() {
    for(unsigned int l = 0; l < agent.num_of_layers(); l++) {
        for(unsigned int n = 0; n < agent.layer(l)->out_features(); n++) {
            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++)
                target.layer(l)->node(n)->set_weight(i, agent.layer(l)->node(n)->weight(i));
        }
    }
}

std::vector<double> DQN::agent_performance(std::vector<std::vector<double>> &state, std::vector<std::vector<double>> &reward, double GAMMA) {
    double mean_loss = 0.00, mean_reward = 0.00;
    for(unsigned int frame = 0; frame < state.size(); frame++) {
        std::vector<double> agent_q_value = agent.predict(state[frame]);
        unsigned int action = std::max_element(agent_q_value.begin(), agent_q_value.end()) - agent_q_value.begin();

        double expected_reward = reward[frame][action];
        if(frame != state.size() - 1) {
            std::vector<double> target_q_value = target.predict(state[frame+1]);
            expected_reward += *std::max_element(target_q_value.begin(), target_q_value.end()) * GAMMA;

            std::vector<double>().swap(target_q_value);
        }

        mean_loss += pow(expected_reward - agent_q_value[action], 2);
        mean_reward += reward[frame][action];

        std::vector<double>().swap(agent_q_value);
    }

    mean_loss /= state.size();
    mean_reward /= state.size();

    return std::vector<double>({mean_loss, mean_reward});
}

void DQN::optimize(std::vector<std::vector<double>> &state, std::vector<std::vector<double>> &reward,
                  unsigned int ITERATION, unsigned int BATCH_SIZE, double ALPHA, double ALPHA_DECAY, double EPSILON, double EPSILON_DECAY, double GAMMA, unsigned int SYNC_FREQUENCY) {
    std::vector<unsigned int> action_memory;
    for(unsigned int frame = 0; frame < state.size(); frame++) {
        EPSILON *= pow(EPSILON_DECAY, frame);
        if(EPSILON < 0.05)
            EPSILON = 0.05;
        // e-greedy policy
        unsigned int action;
        double explore = (double)rand() / RAND_MAX;
        if(explore < EPSILON)
            action = rand() % agent.layer(agent.num_of_layers() - 1)->out_features();
        else {
            std::vector<double> agent_q_value = agent.predict(state[frame]);
            action = std::max_element(agent_q_value.begin(), agent_q_value.end()) - agent_q_value.begin();

            std::vector<double>().swap(agent_q_value);
        }

        action_memory.push_back(action);

        // mini-batch learning (enough replay memory)
        if(action_memory.size() > BATCH_SIZE) {
            std::vector<unsigned int> batch(action_memory.size(), 0);
            std::iota(batch.begin(), batch.end(), 0);
            std::shuffle(batch.begin(), batch.end(), seed);
            batch.erase(batch.begin() + BATCH_SIZE, batch.end());

            ALPHA *= pow(ALPHA_DECAY, frame - BATCH_SIZE);
            if(ALPHA < 0.00001)
                ALPHA = 0.00001;

            for(unsigned int itr = 1; itr <= ITERATION; itr++) {
                for(unsigned int k = 0; k < BATCH_SIZE; k++) {
                    unsigned int index = batch[k];
                    // compute expected reward (finite bellman equation)
                    double expected_reward = reward[index][action_memory[index]];
                    if(index != state.size() - 1) {
                        std::vector<double> target_q_value = target.predict(state[index+1]);
                        expected_reward += *std::max_element(target_q_value.begin(), target_q_value.end()) * GAMMA;

                        std::vector<double>().swap(target_q_value);
                    }

                    // SGD
                    std::vector<double> agent_q_value = agent.predict(state[index]);
                    for(int l = agent.num_of_layers() - 1; l >= 0; l--) {
                        unsigned int start = 0, end = agent.layer(l)->out_features();
                        if(l == agent.num_of_layers() - 1) {
                            start = action_memory[index];
                            end = start + 1;
                        }

                        double partial_gradient = 0.00, gradient = 0.00;
                        for(unsigned int n = start; n < end; n++) {
                            if(l == agent.num_of_layers() - 1)
                                partial_gradient = -2.00 * (expected_reward - agent_q_value[n]);
                            else {
                                partial_gradient = agent.layer(l)->node(n)->err() * relu_prime(agent.layer(l)->node(n)->sum());

                                double updated_bias = agent.layer(l)->node(n)->bias() - ALPHA * partial_gradient;
                                agent.layer(l)->node(n)->set_bias(updated_bias);
                            }

                            for(unsigned int i = 0; i < agent.layer(l)->in_features(); i++) {
                                if(l == 0)
                                    gradient = partial_gradient * state[index][i];
                                else {
                                    gradient = partial_gradient * agent.layer(l-1)->node(i)->act();
                                    agent.layer(l-1)->node(i)->add_err(partial_gradient * agent.layer(l)->node(n)->weight(i));
                                }

                                double updated_weight = agent.layer(l)->node(n)->weight(i) - ALPHA * gradient;
                                agent.layer(l)->node(n)->set_weight(i, updated_weight);
                            }
                        }
                    }

                    std::vector<double>().swap(agent_q_value);
                }
            }

            std::vector<double> results = agent_performance(state, reward, GAMMA);
            double mean_loss = results[0], mean_reward = results[1];

            progress_bar(frame, state.size(), "(frame=" + std::to_string(frame) + ") L = " + std::to_string(mean_loss) + ", R = " + std::to_string(mean_reward));

            std::ofstream out("./data/training_performance", std::ios_base::app);
            out << std::to_string(mean_loss) << " " << std::to_string(mean_reward) << "\n";
            out.close();

            std::vector<double>().swap(results);
            std::vector<unsigned int>().swap(batch);

            if(frame - BATCH_SIZE % SYNC_FREQUENCY == 0)
                sync();
        }
    }

    std::vector<unsigned int>().swap(action_memory);
}

