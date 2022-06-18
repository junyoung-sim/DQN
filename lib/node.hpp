#ifndef __NODE_HPP_
#define __NODE_HPP_

#include <vector>

class Node
{
private:
    double b; // bias
    double s; // sum
    double z; // activation
    double e; // error
    std::vector<double> w;
public:
    Node() {}
    Node(unsigned int in) {
        init();
        b = 0.00;
        for(unsigned int i = 0; i < in; i++)
            w.push_back(0.00);
    }
    ~Node() {
        std::vector<double>().swap(w);
    }

    double bias();
    double sum();
    double act();
    double err();
    double weight(unsigned int index);

    void init();
    void update_bias(double gradient);
    void set_sum(double val);
    void set_act(double val);
    void add_err(double val);
    void set_weight(unsigned int index, double val);
};

#endif
