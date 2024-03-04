#include <layer/base_structure_layer.h>

namespace ATP{

template <class value_type>
void base_structure_t<value_type>::increase_dy_use_counter( ) {
	for (int i = 0; i < dy.size(); i++) {
		dy[i]->increase_use_counter(BACKWARD);
        // inputs[i]->printTensorState("input");
	}
}

template <class value_type>
void base_structure_t<value_type>::increase_dy_cur_use_counter( ) {
	for (int i = 0; i < dy.size(); i++) {
		dy[i]->increase_cur_use_counter(BACKWARD);
        // inputs[i]->printTensorState("input");
	}
}

template <class value_type>
void base_structure_t<value_type>::update_dy_state( ) {
    for (int i = 0; i < dy.size(); i++) {
        if (dy[i]->is_tensor_useless(BACKWARD)) {
            dy[i]->set_data_state(NO_COMPUTED);
            dy[i]->set_data_position(NO_DATA);
            dy[i]->reset_cur_use_counter(BACKWARD);
        }
    }
}

template <class value_type>
void base_structure_t<value_type>::increase_input_use_counter(net_comp dir) {
	for (int i = 0; i < inputs.size(); i++) {
		inputs[i]->increase_use_counter(dir);
        // inputs[i]->printTensorState("input");
	}
}

template <class value_type>
void base_structure_t<value_type>::increase_input_cur_use_counter(net_comp dir) {
	for (int i = 0; i < inputs.size(); i++) {
		inputs[i]->increase_cur_use_counter(dir);
        // inputs[i]->printTensorState("input");
	}
}

template <class value_type>
void base_structure_t<value_type>::update_output_state(net_comp dir) {
    if (dir == FORWARD) {
        // for (int i = 0; i < inputs.size(); i++) {
        //     if (inputs[i]->is_tensor_useless(FORWARD)) {
        //         inputs[i]->set_data_state(FORWARD_DELETE_OK);
        //         inputs[i]->reset_cur_use_counter(FORWARD);
        //     }
        // }
    }
    else if (dir == BACKWARD) {
        // for (int i = 0; i < inputs.size(); i++) {
        //     if (inputs[i]->is_tensor_useless(BACKWARD)) {
        //         inputs[i]->set_data_state(NO_COMPUTED);
        //         inputs[i]->set_data_position(NO_DATA);
        //         inputs[i]->reset_cur_use_counter(BACKWARD);
        //     }
        // }
    }
}

template <class value_type>
void base_structure_t<value_type>::update_input_state(net_comp dir) {
    if (dir == FORWARD) {
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs[i]->is_tensor_useless(FORWARD)) {
                inputs[i]->set_data_state(FORWARD_DELETE_OK);
                inputs[i]->reset_cur_use_counter(FORWARD);
            }
        }
    }
    else if (dir == BACKWARD) {
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs[i]->is_tensor_useless(BACKWARD)) {
                inputs[i]->set_data_state(NO_COMPUTED);
                inputs[i]->set_data_position(NO_DATA);
                inputs[i]->reset_cur_use_counter(BACKWARD);
            }
        }
    }
}

INSTANTIATE_CLASS(base_structure_t);
    
} //ATP namespace
