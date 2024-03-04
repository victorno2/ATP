#if !defined(_BASE_LAYER_H_)
#define _BASE_LAYER_H_
#include <vector>
#include <tensor.h>
#include <registry.h>
#include <util/common.h>
#include <solver.h>

namespace ATP{
    
typedef enum LAYER_POSITION {
    REMAIN_LAYER	= 0,
	SWAP_LAYER  	= 1,
    RECOMPUTE_LAYER = 2
} LAYER_POSITION;

typedef enum LAYER_STRUCTURE {
	SISO 	= 0,
	MIMO	= 1
} LAYER_STRUCTURE;

template <class value_type>
class base_layer_t
{
private:
    static size_t instance_counter;
    
    const int id_;
    size_t fcounter = 0; //forward counter,  to assist join
    size_t bcounter = 0; //backward counter, to assist fork
    
    std::vector<base_layer_t*> next_layers;
    std::vector<base_layer_t*> prev_layers;
    
    LAYER layer_type;

	LAYER_STRUCTURE layer_structure;

    LAYER_POSITION layer_position = REMAIN_LAYER;

public:
    
    base_layer_t(LAYER lt):id_(instance_counter) {
        instance_counter++;
        this->layer_type = lt;
    }

    inline LAYER_POSITION get_layer_position() {
		return this->layer_position;
	}

    inline void set_layer_position(LAYER_POSITION layer_position) {
		this->layer_position = layer_position;
	}

	inline LAYER_STRUCTURE get_layer_structure() {
		return this->layer_structure;
	}
	
    inline void set_layer_structure(LAYER_STRUCTURE layer_structure) {
        this->layer_structure = layer_structure;
    }

    //to be implemented at each specific network/structural layers
    virtual std::vector<value_type> forward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg)  = 0;
    virtual void backward(network_stage stage, cublasHandle_t *cublas_h, cudnnHandle_t *cudnn_h, registry_t<value_type> *reg) = 0;
    virtual void forward_setup(registry_t<value_type>* reg = NULL, cudnnHandle_t* cudnn_h = NULL)  = 0;
    virtual void backward_setup(registry_t<value_type>* reg = NULL, cudnnHandle_t* cudnn_h = NULL) = 0;
    virtual void update( cublasHandle_t *cublas_h, size_t iter, base_solver_t<value_type>* solver) = 0;
	virtual void grad_zero(cublasHandle_t *cublas_h, base_solver_t<value_type>* solver) = 0;
    virtual void gen_description(char* buff, size_t* len_in_byte) = 0;

    virtual void fake_run(net_comp dir, registry_t<value_type>* reg) = 0;

	virtual void increase_dy_use_counter() = 0;
	virtual void increase_dy_cur_use_counter() = 0;
	virtual void update_dy_state() = 0;
	
	virtual void increase_input_use_counter(net_comp dir) = 0;
	virtual void increase_input_cur_use_counter(net_comp dir) = 0;
	virtual void update_input_state(net_comp dir) = 0;
	virtual void update_output_state(net_comp dir) = 0;
	// virtual void reset_input_cur_use_counter(net_comp dir);
	
    /**
     * value_type: layer_id
     * value_type: layer_type
     * value_type: previous layers count
     * value_type: next layers count
     * value_type: [pre layer id...], [next layer id...]
     */
    void _gen_meta_description(char* buff, size_t* len_in_byte) {
        size_t SIZE = sizeof(value_type);

        value_type layer_id = id_;
        value_type type = layer_type;
        value_type pre_cnt = prev_layers.size();
        value_type next_cnt = next_layers.size();

        memcpy(buff, &layer_id, SIZE);
        memcpy(buff+SIZE, &type, SIZE);
        memcpy(buff+2*SIZE, &pre_cnt, SIZE);
        memcpy(buff+3*SIZE, &next_cnt, SIZE);

        for (size_t i = 0; i < prev_layers.size(); ++i) {
            value_type tmp = prev_layers[i]->get_base_id();
            memcpy(buff+4*SIZE+i*SIZE, &tmp, SIZE);
        }
        for (size_t i = 0; i < next_layers.size(); ++i) {
            value_type tmp = next_layers[i]->get_base_id();
            memcpy(buff+4*SIZE+prev_layers.size()*SIZE+i*SIZE, &tmp, SIZE);
        }

        *len_in_byte = (4+prev_layers.size()+next_layers.size())*SIZE;

    }
    
    inline std::vector<base_layer_t*> get_next() {
        return next_layers;
    }
	
	inline std::vector<base_layer_t*>* get_next_ptr() {
        return &next_layers;
    }
    
    inline std::vector<base_layer_t*> get_prev() {
        return prev_layers;
    }
    
    //bi-direction hook
    void hook(base_layer_t* c) {
        if(c == NULL) return;
        printf("hook layer %p:%d <-> %p: layer%d-type%d\n", this, id_, c, c->get_base_id(), c->get_layer_type());
        next_layers.push_back(c);
        c->prev_layers.push_back(this);
    }
    
    void clear_prev() {
        printf("before clear prev_layers.size() = %d\n", prev_layers.size());
        while (prev_layers.size() != 0) {
            this->prev_layers.pop_back();
            printf("prev_layers.size() = %d\n", prev_layers.size());
        }
    }

    void switch_prev_l_to(base_layer_t* c) {
        if(c == NULL) return;
        assert(prev_layers.size() == 1);
        this->prev_layers[0] = c;
#ifdef DEBUG
        printf("switch layer %p:%d <- %p:%d\n", prev_layers[0], prev_layers[0]->get_base_id(), c, c->get_base_id() );
        printf("prev layer now:%d\n", prev_layers[0]->get_base_id() );
#endif
    }
    
    //one direction hook
    void hook_to(base_layer_t* c) {
        if(c == NULL) return;
        printf("hook layer %p:%d -> %p:%d\n", this, id_, c, c->get_base_id());
        next_layers.push_back(c);
    }
    
    inline size_t get_fcounter() {
        return fcounter;
    }
    
    inline size_t get_prev_size() {
        return prev_layers.size();
    }
    
    inline size_t get_next_size() {
        return next_layers.size();
    }
    
    inline int get_base_id() {
        return this->id_;
    }
    
    inline void fcounter_inc() {
        this->fcounter = this->fcounter + 1;
    }
    
    inline void reset_fc_counter() {
        this->fcounter = 0;
    }
    
    inline LAYER get_layer_type() {
        return this->layer_type;
    }
    

};

} //SuperNeuron namespace

#endif // _BASE_NETWORK_LAYER_H_


