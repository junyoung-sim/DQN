
#include <cstdlib>
#include <vector>

#include "../lib/node.hpp"

Node::Node(unsigned int in) {
    init();
    b = 0.00;
    for(unsigned int i = 0; i < in; i++)
        w.push_back(0.00);
}

Node::~Node() {
    std::vector<double>().swap(w);
}

double Node::bias() {
    return b;
}

double Node::sum() {
    return s;
}

double Node::act() {
    return z;
}

double Node::err() {
    return e;
}

double Node::weight(unsigned int index) {
    return w[index];
}

void Node::init() {
    s = 0.00;
    z = 0.00;
    e = 0.00;
}

void Node::update_bias(double gradient) {
    b -= gradient;
}

void Node::set_sum(double val) {
    s = val;
}

void Node::set_act(double val) {
    z = val;
}

void Node::add_err(double val) {
    e += val;
}

void Node::set_weight(unsigned int index, double val) {
    w[index] = val;
}

