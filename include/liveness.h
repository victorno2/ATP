//
// Created by ay27 on 9/14/17.
//

#ifndef ATP_LIVENESS_H
#define ATP_LIVENESS_H

#include <util/common.h>
#include <tensor.h>
#include <registry.h>
#include <layer/base_layer.h>

namespace ATP {


template<class value_type>
class liveness_analysis_t {
private:

    size_t total_size = 0;
	std::map<int, size_t> f_mem_usage;    // layer_id & size
    std::map<int, size_t> b_mem_usage;    // layer_id & size
	std::map<int, size_t> f_ouput_usage;    // layer_id & size
    std::map<int, size_t> b_ouput_usage;    // layer_id & size
	size_t bottleneck_mem_size;
	size_t bottleneck_output_size;
    size_t bottleneck_data_size;
	size_t max_grad_size;

    std::vector<std::vector<void *> > f_stash_tensors;
    std::vector<std::vector<void *> > b_stash_tensors;
    std::vector<std::vector<void *> > f_free_tensors;
    std::vector<std::vector<void *> > b_free_tensors;
    registry_t <value_type> *reg;
    std::map<void *, mem_mode> *regulated_tensors;

    std::vector<std::vector<std::pair<int, net_comp> > > *subsequent_forward;
    std::vector<std::vector<std::pair<int, net_comp> > > *subsequent_backward;

    std::vector<LAYER> CHECKPOINT_LAYERS;

    int max_layer_id;

    inline std::vector<std::pair<int, net_comp> >& get_subsequent_layers(int curt_layer_id, net_comp dir);

    bool is_used_by_layer(int layer_id, net_comp dir, tensor_t <value_type> *t);

    bool is_freeable_afterwards(int curt_layer_id, net_comp dir, tensor_t <value_type> *t);

    void set_ins(std::vector<std::vector<void *> > *ins, net_comp dir);

    void set_outs(std::vector<std::vector<void *> > *outs, net_comp dir);

    inline bool is_checkpoint(base_layer_t <value_type> *l) {
        for (auto it = CHECKPOINT_LAYERS.begin(); it != CHECKPOINT_LAYERS.end(); ++it) {
            if (l->get_layer_type() == *it) {
                return true;
            }
            if (l->get_layer_type() == DATA_L || l->get_layer_type() == SOFTMAX) {
				// 数据层和SOFTMAX层一定是CHECKPOINT_LAYERS
                return true;
            }
        }
        return false;
    }


public:
    liveness_analysis_t(registry_t <value_type> *_reg, std::map<void *, mem_mode> *_regulated_tensors,
                        std::vector<std::vector<std::pair<int, net_comp>>> *_subsequent_forward,
                        std::vector<std::vector<std::pair<int, net_comp>>> *_subsequent_backward,
                        const std::vector<LAYER> &_CHECKPOINT_LAYERS,
                        int _max_layer_id) : reg(_reg), regulated_tensors(_regulated_tensors),
                                             subsequent_forward(_subsequent_forward),
                                             subsequent_backward(_subsequent_backward),
                                             CHECKPOINT_LAYERS(_CHECKPOINT_LAYERS),
                                             max_layer_id(_max_layer_id) {
        printf("\n\n");
        // set_ins(&f_stash_tensors, FORWARD);
        // set_ins(&b_stash_tensors, BACKWARD);
        // set_outs(&f_free_tensors, FORWARD);
        // set_outs(&b_free_tensors, BACKWARD);
		find_bottleneck();

// #define DEBUG
#ifdef DEBUG
        printf("--------f_stash_tensors-----------\n");
        for (int layer_id = 1; layer_id < (int)f_stash_tensors.size(); ++layer_id) {
            printf("layer : %d\n", layer_id);
            for (size_t i = 0; i < f_stash_tensors[layer_id].size(); ++i) {
                tensor_t<value_type>* t = (tensor_t<value_type>*)f_stash_tensors[layer_id][i];
                printf("tensor %p : layer %d, type %d, state %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
            }
        }
        printf("\n\n");
        printf("--------b_stash_tensors-----------\n");
        for (int layer_id = 1; layer_id < (int)b_stash_tensors.size(); ++layer_id) {
            printf("layer : %d\n", layer_id);
            for (size_t i = 0; i < b_stash_tensors[layer_id].size(); ++i) {
                tensor_t<value_type>* t = (tensor_t<value_type>*)b_stash_tensors[layer_id][i];
                printf("tensor %p : layer %d, type %d, state %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
            }
        }
        printf("\n\n");
        printf("--------f_free_tensors-----------\n");
        for (int layer_id = 1; layer_id < (int)f_free_tensors.size(); ++layer_id) {
            printf("layer : %d\n", layer_id);
            for (size_t i = 0; i < f_free_tensors[layer_id].size(); ++i) {
                tensor_t<value_type>* t = (tensor_t<value_type>*)f_free_tensors[layer_id][i];
                printf("tensor %p : layer %d, type %d, state %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
            }
        }
        printf("\n\n");
        printf("--------b_free_tensors-----------\n");
        for (int layer_id = 1; layer_id < (int)b_free_tensors.size(); ++layer_id) {
            printf("layer : %d\n", layer_id);
            for (size_t i = 0; i < b_free_tensors[layer_id].size(); ++i) {
                tensor_t<value_type>* t = (tensor_t<value_type>*)b_free_tensors[layer_id][i];
                printf("tensor %p : layer %d, type %d, state %d\n", t, t->get_layer_id(), t->get_type(), t->get_state());
            }
        }
        printf("\n\n");
#endif
// #undef DEBUG
    }

    void reset_liveness_analysis_t(registry_t <value_type> *_reg, std::map<void *, mem_mode> *_regulated_tensors,
                        std::vector<std::vector<std::pair<int, net_comp>>> *_subsequent_forward,
                        std::vector<std::vector<std::pair<int, net_comp>>> *_subsequent_backward,
                        const std::vector<LAYER> &_CHECKPOINT_LAYERS,
                        int _max_layer_id) {

        this->reg = _reg;
        this->regulated_tensors = _regulated_tensors;
        this->subsequent_forward = _subsequent_forward;
        this->subsequent_backward = _subsequent_backward;
        this->CHECKPOINT_LAYERS = _CHECKPOINT_LAYERS;
        this->max_layer_id = _max_layer_id;
        
        // set_ins(&f_stash_tensors, FORWARD);
        // set_ins(&b_stash_tensors, BACKWARD);
        // set_outs(&f_free_tensors, FORWARD);
        // set_outs(&b_free_tensors, BACKWARD);
		find_bottleneck();
    }

	void find_bottleneck();
	
    size_t get_total_size_v2();
	
	size_t get_total_size();

	size_t get_bottleneck_size() {
		return this->bottleneck_mem_size;
	}
	
	size_t get_max_grad_size() {
		return this->max_grad_size;
	}
	
	size_t get_bottleneck_output_size() {
		return this->bottleneck_output_size;
	}

    size_t get_bottleneck_data_size() {
		return this->bottleneck_data_size;
	}

    void clear_related_record() {
        while(f_stash_tensors.size() != 0) {
            auto v = &f_stash_tensors.back();
            while(v->size() != 0) {
                v->pop_back();
            }
            f_stash_tensors.pop_back();
        }
		f_stash_tensors.erase(f_stash_tensors.begin(), f_stash_tensors.end());
        // printf("clear f_stash_tensors done\n");
        while(b_stash_tensors.size() != 0) {
            auto v = &b_stash_tensors.back();
            while(v->size() != 0) {
                v->pop_back();
            }
            b_stash_tensors.pop_back();
        }
		b_stash_tensors.erase(b_stash_tensors.begin(), b_stash_tensors.end());
        // printf("clear b_stash_tensors done\n");
        while(f_free_tensors.size() != 0) {
            auto v = &f_free_tensors.back();
            while(v->size() != 0) {
                v->pop_back();
            }
            f_free_tensors.pop_back();
        }
		f_free_tensors.erase(f_free_tensors.begin(), f_free_tensors.end());
        // printf("clear f_free_tensors done\n");
        while(b_free_tensors.size() != 0) {
            auto v = &b_free_tensors.back();
            while(v->size() != 0) {
                v->pop_back();
            }
            b_free_tensors.pop_back();
        }
		b_free_tensors.erase(b_free_tensors.begin(), b_free_tensors.end());
        // printf("clear b_free_tensors done\n");
    }

    void stash(int layer_id, net_comp dir);

    void update(int layer_id, net_comp dir);

	std::vector<std::vector<void *> > *get_f_stash_tensors() {
		return &f_stash_tensors;
	}

	std::vector<std::vector<void *> > *get_b_stash_tensors() {
		return &b_stash_tensors;
	}

    std::vector<std::vector<void *> > *get_f_free_tensors() {
		return &f_free_tensors;
	}

	std::vector<std::vector<void *> > *get_b_free_tensors() {
		return &b_free_tensors;
	}
};


} // namespace ATP

#endif //ATP_LIVENESS_H
