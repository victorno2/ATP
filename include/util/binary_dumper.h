//
// Created by ay27 on 7/20/17.
//

#ifndef ATP_BINARY_DUMPER_H
#define ATP_BINARY_DUMPER_H

#include <util/common.h>
#include <util/base_reader.h>

namespace ATP {

class Dumper {
private:

    void write_meta();

    uint64_t n, c, h, w;
    std::ofstream file;

    const char *dst_path;

public:
    Dumper(size_t n, size_t c, size_t h, size_t w, const char *dst_path, bool append = false);

    ~Dumper();

    void dump_image(const char *src_data, size_t data_cnt);

    void dump_label(const char *src_data, size_t data_cnt);

    void dump_label(const label_t *src_data, size_t data_cnt);

    void fix_N(size_t n);
};

} // namespace ATP

#endif //ATP_BINARY_DUMPER_H
