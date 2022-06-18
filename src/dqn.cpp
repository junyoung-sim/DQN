
#include <cstdlib>
#include <vector>

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
    std::vector<double> reward_memory;

    for(unsigned int t = 0; t < state.size() - 1; t++) {
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
        reward_memory.push_back(reward[t][action_t]);
        memory_size++;
/*
        // learning phase
        if(memory_size >= BATCH_SIZE * 2) {
            std::vector<unsigned int> batch_index;
            for(unsigned int k = 0; k < BATCH_SIZE; k++) {
                unsigned int index;
                while(std::find(batch_index.begin(), batch_index.end(), index) == batch_index.end())
                    index = rand() % memory_size;

                std::vector<double> target_q = target.predict(state[index+1]);
                double y = reward_memory[index] + GAMMA * (*std::max_element(target_q.begin(), target_q.end()));
            }
        }
*/
    }
}
