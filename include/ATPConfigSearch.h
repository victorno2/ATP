
//#ifndef ATP_GA_H
//#define ATP_GA_H

#include <stdlib.h>
// #include <ATPSearch.h>
#include <ATP.h>
#include <ga.h>

namespace ATP {

void ThroughputPeakSearch( base_layer_t<float>* first_layer, base_layer_t<float>* last_layer, 
                network_t<float>* net, preprocessor<float> *p, parallel_reader_t<float>* reader,
                size_t gpu_mem_size, double pcie_bandwidth, size_t extra_size,
                int start_b, int end_b, int step, int iter_times, int no_update_win, bool is_seq,
                int pop_size, double mut_prob, int ga_iter_times) {
    ATPSearch<float> *apt_net = new ATPSearch<float>(net);
    apt_net->set_simulator(gpu_mem_size, pcie_bandwidth);
    apt_net->set_inherent_size(extra_size);
    size_t best_batch_size;
    double max_tp = 0;
    int no_updata_times = 0;
    double cur_tp = 0;
    for (int batch_size = start_b; batch_size <= end_b; batch_size += step) {
        p->clear_processors();
        ((data_layer_t<float>*)first_layer)->ResetBatchSize(batch_size, is_seq);
        apt_net->set_simulator_trainning(p, reader, net, batch_size, first_layer, last_layer, NULL, NULL, NULL);
        printf("after set_simulator_trainning mem_size = %f\n", BYTE_TO_MB(query_used_mem()));
        if (apt_net->simulate_trainning(net, batch_size, iter_times)) {

        }
        else {
            break;
        }
        cur_tp = apt_net->GetThroughput(batch_size);
        if (cur_tp > max_tp) {
            max_tp = cur_tp;
            best_batch_size = batch_size;
            no_updata_times = 0;
        }
        else {
            no_updata_times++;
        }
        if (no_updata_times > no_update_win) {
            break;
        }
    }
    printf("Best batch size = %d, max_tp = %f\n", best_batch_size, max_tp);
    p->clear_processors();
    ((data_layer_t<float>*)first_layer)->ResetBatchSize(best_batch_size, is_seq);
    apt_net->set_simulator_trainning(p, reader, net, best_batch_size, first_layer, last_layer, NULL, NULL, NULL);
    printf("after set_simulator_trainning mem_size = %f\n", BYTE_TO_MB(query_used_mem()));
    apt_net->simulate_trainning(net, best_batch_size, iter_times);
    int ind_size = net->get_max_layer_id();
    ga apt_ga(pop_size, ind_size, mut_prob, apt_net);
    apt_ga.evolution(ga_iter_times, 0);
    exit(0);
}

}

// #endif //ATP_ATP_ConfigSearch_H