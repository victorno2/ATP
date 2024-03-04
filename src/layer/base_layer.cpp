#include <layer/base_layer.h>

namespace ATP{
    
template <typename value_type>
size_t base_layer_t<value_type>::instance_counter = 1;

INSTANTIATE_CLASS(base_layer_t);

} //ATP namespace
