
#include <cstdlib>
#include <vector>
#include <string>

#include "../lib/layer.hpp"

Layer::Layer(unsigned int _in, unsigned int _out): in(_in), out(_out) {
    for(unsigned int i = 0; i < out; i++)
        n.push_back(Node(in));
}

Layer::~Layer() {
    std::vector<Node>().swap(n);
}

Node *Layer::node(unsigned int index) {
    return &n[index];
}

unsigned int Layer::in_features() {
    return in;
}

unsigned int Layer::out_features() {
    return out;
}

