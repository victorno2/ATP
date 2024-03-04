#if !defined(_BASE_STRUCTURE_LAYER_H_)
#define _BASE_STRUCTURE_LAYER_H_
#include <tensor.h>
#include <registry.h>
#include <util/common.h>
#include <layer/base_layer.h>

namespace ATP{
    
template <class value_type>
class base_structure_t:base_layer_t<value_type>
{
private:
    std::vector<tensor_t<value_type>* > inputs;  //JOIN layer inputs
    std::vector<tensor_t<value_type>* > outputs; //FORK layer outputs
    std::vector<tensor_t<value_type>* > b_data;  //FORK layer has one, while JOIN has multiples
    std::vector<tensor_t<value_type>* > dy; 
	structure_type type;
    
public:
    
    base_structure_t(LAYER lt):base_layer_t<value_type>(lt) {
        this->set_layer_structure(MIMO);
	}
    
    inline int get_id() {
        return this->get_base_id();
    }
    
    virtual std::vector<value_type> forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg)  = 0;
    virtual void backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg) = 0;
    virtual void forward_setup(registry_t<value_type>* reg = NULL, cudnnHandle_t* cudnn_h = NULL)  = 0;
    virtual void backward_setup(registry_t<value_type>* reg = NULL, cudnnHandle_t* cudnn_h = NULL) = 0;
    virtual void gen_description(char* buff, size_t* len_in_byte) = 0;
    virtual void fake_run(net_comp dir, registry_t<value_type> *reg) = 0;

	void increase_dy_use_counter();
	void increase_dy_cur_use_counter();
	void update_dy_state();
	
	void increase_input_use_counter(net_comp dir);
	void increase_input_cur_use_counter(net_comp dir);
	void update_input_state(net_comp dir);
	void update_output_state(net_comp dir);
	/*
	void increase_input_cur_use_counter(net_comp dir) {
		for (int i = 0; i < inputs.size(); i++) {
			inputs[i]->increase_cur_use_counter(dir);
		}
	}
	void reset_input_cur_use_counter(net_comp dir) {
		for (int i = 0; i < inputs.size(); i++) {
			inputs[i]->reset_cur_use_counter(dir);
		}
	}
	*/
    void gen_meta_description(char* buff, size_t* len_in_byte) {
        this->_gen_meta_description(buff, len_in_byte);
    }

    void update(cublasHandle_t *cublas_h, size_t iter, base_solver_t<value_type>* solver ) {
    //structure layer do not update
    };
    
	void grad_zero(cublasHandle_t *cublas_h, base_solver_t<value_type>* solver) {
		//structure layer has not grad
	};
	
    std::vector<std::pair<int, int> > get_inputs_keys() {
        std::vector<base_layer_t<value_type>* > prev_layers = this->get_prev();
        int curt_l_id = this->get_base_id();
        std::vector<std::pair<int, int> > result;
        
        for(size_t i = 0; i < prev_layers.size(); i++) {
            int prev_id = prev_layers[i]->get_base_id();
            d_key_t key(prev_id, curt_l_id);
            result.push_back(key);
        }
        //verification
        if (type == FORK_L) {
            assert(result.size() <= 1);
        } else {
            assert(result.size() >= 1);
        }
        return result;
    }
    
    std::vector<std::pair<int, int> > get_outputs_keys() {
        std::vector<base_layer_t<value_type>* > next_layers = this->get_next();
        int curt_l_id = this->get_base_id();
        std::vector<std::pair<int, int> > result;
        
        if(next_layers.size() == 0) {
            d_key_t key(curt_l_id, 0);
            result.push_back(key);
        } 
		else {
            for(size_t i = 0; i < next_layers.size(); i++) {
                int next_id = next_layers[i]->get_base_id();
                d_key_t key(curt_l_id, next_id);
                result.push_back(key);
            }
        }
        
        //verification
        if (type == FORK) {
            assert(result.size() >= 1);
        } else {
            assert(result.size() <= 1);
        }
        return result;
    }
    
    void clear_inputs() {
		inputs.erase(inputs.begin(), inputs.end());
        // while (inputs.size() != 0) {
        //     inputs.pop_back();
        // }
    }

    void clear_outputs() {
		outputs.erase(outputs.begin(), outputs.end());
        // while (outputs.size() != 0) {
        //     outputs.pop_back();
        // }
    } 

    void clear_b_data() {
		b_data.erase(b_data.begin(), b_data.end());
        // while (b_data.size() != 0) {
        //     b_data.pop_back();
        // }
    }

	std::vector<tensor_t<value_type>* > get_dy() {
        if (type == FORK || type == JOIN) {
            assert(dy.size()  >= 1);
            return this->dy;
        } else if(type == JOIN) {
            assert(dy.size()  == 1);
            return this->dy;
        }
        return std::vector<tensor_t<value_type>* >();
    }

	void clear_dy() {
		dy.erase(dy.begin(), dy.end());
        // while (b_data.size() != 0) {
        //     b_data.pop_back();
        // }
    }
	
	void set_dy(tensor_t<value_type>* t) {
		if(type == JOIN || type == FORK) {
            dy.push_back(t);
            assert(inputs.size() <= 1);
        } else {
            inputs.push_back(t);
        }
	}

    void set_input(tensor_t<value_type>* t) {
        //the input a FORK layer has to be one
		// inputs.swap(std::vector<tensor_t<value_type>*>()); 
		// std::vector<tensor_t<value_type>*>().swap(inputs);
        if(type == FORK) {
            inputs.push_back(t);
			// printf("after fork set_input, inputs_size() = %d\n", inputs.size());
            assert(inputs.size() <= 1);
        } else {
            inputs.push_back(t);
        }
        // printf("@layer%d curt_input size %lu\n", get_id(), inputs.size() );
        if(inputs.size() == 0) {
            return;
        } else {
            tensor_t<value_type>* tmp = inputs[0];
            assert(tmp->get_N() == t->get_N());
            // we don't check the channel size
            // assert(tmp->get_C() == t->get_C());
            assert(tmp->get_H() == t->get_H());
            assert(tmp->get_W() == t->get_W());
        }
		// printf("after set_input, inputs_size() = %d\n", inputs.size());
    }
    
    //by convention, each layer only holds the output tensors
    void set_output(tensor_t<value_type>* t, std::pair<int, int> idx, registry_t<value_type>* reg) {
        // the output of a FORK layer has to be one
		// outputs.swap(std::vector<tensor_t<value_type>*>()); 
		// std::vector<tensor_t<value_type>*>().swap(outputs);
        if(type == JOIN || type == CONCAT) {
            outputs.push_back(t);
            assert(outputs.size() <= 1);
        } else {
            outputs.push_back(t);
        } 
        reg->register_output(idx.first, idx.second, t);
        // printf("@layer%d curt_output size %lu\n", get_id(), inputs.size() );
    } 
    
    void set_b_data(tensor_t<value_type>* t, std::pair<int, int> idx, registry_t<value_type>* reg) {
        // the output of a FORK layer has to be one
		// b_data.swap(std::vector<tensor_t<value_type>*>()); 
		// std::vector<tensor_t<value_type>*>().swap(b_data);
        if(type == JOIN || type == CONCAT) {
            b_data.push_back(t);
            assert(b_data.size() >= 1);
            //b data is the reverse of input pair
        } else if(type == FORK) {
            b_data.push_back(t);
            assert(b_data.size() <= 1);
        }
        reg->register_b_data(idx.second, idx.first, t);
        // printf("@layer%d curt_b_data size %lu\n", get_id(), b_data.size() );
    }
    
    void set_structure_type(structure_type t) {
        type = t;
    }
    
    std::vector<tensor_t<value_type>* > get_b_data() {
        if (type == FORK) {
            assert(inputs.size()  <= 1);
            assert(outputs.size() >= 1);
            assert(b_data.size()  <= 1);
            return this->b_data;
        } else if(type == JOIN) {
            assert(inputs.size()  >= 1);
            assert(outputs.size() <= 1);
            assert(b_data.size()  >= 1);
            return this->b_data;
        }
        return std::vector<tensor_t<value_type>* >();
    }
    
    std::vector<tensor_t<value_type>* > get_inputs() {
        if (type == FORK) {
			// printf("get fork inputs.size() = %d\n", inputs.size());
            assert(inputs.size()  <= 1);
            assert(outputs.size() >= 1);
            return this->inputs;
        } else if(type == JOIN) {
            assert(inputs.size()  >= 1);
            assert(outputs.size() <= 1);
            return this->inputs;
        }
        return std::vector<tensor_t<value_type>* >();
    }
	
	std::vector<tensor_t<value_type>* >* get_inputs_ptr() {
        if (type == FORK) {
            assert(inputs.size()  <= 1);
            assert(outputs.size() >= 1);
            return &(this->inputs);
        } else if(type == JOIN) {
            assert(inputs.size()  >= 1);
            assert(outputs.size() <= 1);
            return &(this->inputs);
        }
		return NULL;
    }
    
    std::vector<tensor_t<value_type>* > get_outputs() {
        if (type == FORK) {
            assert(inputs.size()  <= 1);
            assert(outputs.size() >= 1);
            return this->outputs;
        } else if(type == JOIN) {
            assert(inputs.size()  >= 1);
            assert(outputs.size() <= 1);
            return this->outputs;
        }
        return std::vector<tensor_t<value_type>* >();
    }

    std::vector<tensor_t<value_type>* >* get_outputs_ptr() {
        if (type == FORK) {
            assert(inputs.size()  <= 1);
            assert(outputs.size() >= 1);
            return &(this->outputs);
        } else if(type == JOIN) {
            assert(inputs.size()  >= 1);
            assert(outputs.size() <= 1);
            return &(this->outputs);
        }
        return NULL;
    }

};


} // superneuron namespace
#endif // _BASE_STRUCTURE_LAYER_H_
