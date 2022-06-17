#ifndef __NODE_HPP_
#define __NODE_HPP_

#include <vector>

class Node
{
private:
    double b;
    double s;
    double z;
    double e;
    std::vector<double> w;
public:
    Node(){}
    Node(unsigned int in);
    ~Node();

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
