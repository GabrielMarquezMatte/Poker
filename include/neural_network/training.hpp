#ifndef __POKER_TRAINING_HPP__
#define __POKER_TRAINING_HPP__
#include "dlib_net.hpp"
#include <dlib/data_io.h>

void trainSupervised(net_type &net, const std::vector<dlib::matrix<float>> &X, const std::vector<unsigned long> &y)
{
    dlib::dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(1e-3);
    trainer.set_min_learning_rate(1e-5);
    trainer.set_mini_batch_size(128);
    trainer.be_verbose();
    trainer.train(X, y);
    dlib::serialize("policy_supervised.dat") << net;
}

#endif // __POKER_TRAINING_HPP__