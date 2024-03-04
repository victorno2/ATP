#include <layer/base_network_layer.h>

namespace ATP{

template <class value_type>
void base_network_layer_t<value_type>::increase_dy_use_counter( ) {
    if (this->dy != NULL) {
        dy->increase_use_counter(BACKWARD);
    }
}

template <class value_type>
void base_network_layer_t<value_type>::increase_dy_cur_use_counter( ) {
    if (this->dy != NULL) {
        dy->increase_cur_use_counter(BACKWARD);
    }
}

template <class value_type>
void base_network_layer_t<value_type>::update_dy_state( ) {
    if (this->dy != NULL) {
        if (this->dy->is_tensor_useless(BACKWARD)) {
            this->dy->set_data_state(NO_COMPUTED);
            this->dy->set_data_position(NO_DATA);
            this->dy->reset_cur_use_counter(BACKWARD);
        }
    } 
}

template <class value_type>
void base_network_layer_t<value_type>::increase_input_use_counter(net_comp dir) {
    if (this->f_in != NULL) {
        this->f_in->increase_use_counter(dir);
        // f_in->printTensorState("input");
    }
}

template <class value_type>
void base_network_layer_t<value_type>::increase_input_cur_use_counter(net_comp dir) {
    if (this->f_in != NULL) {
        this->f_in->increase_cur_use_counter(dir);
        // f_in->printTensorState("input");
    }
}

template <class value_type>
void base_network_layer_t<value_type>::update_output_state(net_comp dir) {
    if (this->f_in != NULL) {
        // printf("f_out = tensor%d-layer%d gpu_ptr = %x\n", f_out->get_tensor_id(), f_out->get_layer_id(), f_out->get_gpu_ptr());
        if (dir == FORWARD) { 
            if (this->f_out->is_tensor_useless(FORWARD)) {
                this->f_out->set_data_state(FORWARD_DELETE_OK);
                this->f_out->reset_cur_use_counter(FORWARD);
            }
            // if (f_out->get_tensor_id() == 6323) {
            //     f_out->printTensorData("update_input_state t_in", 2);
            // }
        }
        else if (dir == BACKWARD) {
            // if (f_in->get_tensor_id() == 6323) {
            //     f_in->printTensorData("update_input_state t_in", 2);
            // }
            if (this->f_out->is_tensor_useless(BACKWARD)) {
                this->f_out->set_data_state(NO_COMPUTED);
                this->f_out->set_data_position(NO_DATA);
                this->f_out->reset_cur_use_counter(BACKWARD);
            }
        }
    }
}

template <class value_type>
void base_network_layer_t<value_type>::update_input_state(net_comp dir) {
    if (this->f_in != NULL) {
        // printf("f_in = tensor%d-layer%d gpu_ptr = %x\n", f_in->get_tensor_id(), f_in->get_layer_id(), f_in->get_gpu_ptr());
        if (dir == FORWARD) { 
            if (this->f_in->is_tensor_useless(FORWARD)) {
                this->f_in->set_data_state(FORWARD_DELETE_OK);
                this->f_in->reset_cur_use_counter(FORWARD);
            }
        }
        else if (dir == BACKWARD) {
            if (this->f_in->is_tensor_useless(BACKWARD)) {
                this->f_in->set_data_state(NO_COMPUTED);
                this->f_in->set_data_position(NO_DATA);
                this->f_in->reset_cur_use_counter(BACKWARD);
            }
        }
    }
}

INSTANTIATE_CLASS(base_network_layer_t);

} // ATP namespace
