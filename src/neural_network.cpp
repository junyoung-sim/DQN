
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <cmath>

#include "../lib/neural_network.hpp"

double relu(double x) {
    return x > 0.00 ? x : 0.00;
}

double relu_prime(double x) {
    return x > 0.00 ? 1.00 : 0.00;
}

// --- //

NeuralNetwork::~NeuralNetwork() {
    std::vector<Layer>().swap(layers);
}

void NeuralNetwork::add_layer(unsigned int in, unsigned int out) {
    layers.push_back(Layer(in, out));
}

void NeuralNetwork::initialize(std::default_random_engine &seed) {
    std::normal_distribution<double> std_normal(0.0, 1.0);
    // He-initialization
    for(Layer &layer: layers) {
        for(unsigned int n = 0; n < layer.out_features(); n++) {
            for(unsigned int i = 0; i < layer.in_features(); i++)
                layer.node(n)->set_weight(i, std_normal(seed) * sqrt(2.00 / layer.in_features()));
        }
    }
}

Layer *NeuralNetwork::layer(unsigned int index) {
    return &layers[index];
}

std::vector<double> NeuralNetwork::predict(std::vector<double> &x) {
    std::vector<double> yhat;
    // fully-connected feedforward
    for(unsigned int l = 0; l < layers.size(); l++) {
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
           double matmul = 0.00;
            for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                if(l == 0)
                    matmul += x[i] * layers[l].node(n)->weight(i);
                else
                    matmul += layers[l-1].node(i)->act() * layers[l].node(n)->weight(i);
            }

            layers[l].node(n)->init();
            layers[l].node(n)->set_sum(matmul + layers[l].node(n)->bias());

            if(l != layers.size() - 1)
                layers[l].node(n)->set_act(relu(layers[l].node(n)->sum())); // hidden layer (relu)
            else
                yhat.push_back(layers[l].node(n)->sum()); // output layer (linear q-values)
        }
    }

    return yhat;
}

