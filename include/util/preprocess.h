//
// Created by ay27 on 17/6/9.
//

#ifndef ATP_PREPROCESS_H
#define ATP_PREPROCESS_H

#include <tensor.h>
#include <vector>
#include <fstream>

namespace ATP {

template<class value_type>
class base_preprocess_t {
public:
	/*
	virtual void reset(size_t N, size_t C, size_t H, size_t W, size_t padH, size_t padW);
	virtual void reset(size_t src_N, size_t src_C, size_t src_H, size_t src_W,
                  size_t dst_N, size_t dst_C, size_t dst_H, size_t dst_W);
	virtual void reset(size_t N, size_t C, size_t H, size_t W, value_type max_delta, bool is_src_regularized = false);
	virtual void reset(size_t N, size_t C, size_t H, size_t W, value_type lower, value_type upper);
	virtual void reset(size_t N_, size_t C_, size_t H_, size_t W_, const char *mean_file_path, value_type scale = 1.0);
	virtual void reset(size_t N_, size_t C_, size_t H_, size_t W_, value_type* channel_mean, value_type scale = 1.0);
	*/

    virtual void transfer(value_type *src, value_type *dst) = 0;

    virtual size_t output_size() = 0;

    virtual size_t input_size() = 0;
};


template<class value_type>
class border_padding_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W, padH, padW;
public:
    border_padding_t(size_t N, size_t C, size_t H, size_t W, size_t padH, size_t padW) :
            N(N), C(C), H(H), W(W), padH(padH), padW(padW) {
        assert(padH > 0);
        assert(padW > 0);
    }

    void transfer(value_type *src, value_type *dst) override;

	void reset(size_t N, size_t C, size_t H, size_t W, size_t padH, size_t padW) {
		this->N = N;
		this->C = C;
		this->H = H;
		this->W = W;
		this->padH = padH;
		this->padW = padW;
	}

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class random_crop_t : public base_preprocess_t<value_type> {
private:
    size_t src_N, src_C, src_H, src_W, dst_N, dst_C, dst_H, dst_W;
public:
    random_crop_t(size_t src_N, size_t src_C, size_t src_H, size_t src_W,
                  size_t dst_N, size_t dst_C, size_t dst_H, size_t dst_W) :
            src_N(src_N), src_C(src_C), src_H(src_H), src_W(src_W),
            dst_N(dst_N), dst_C(dst_C), dst_H(dst_H), dst_W(dst_W) {
        assert(src_N == dst_N);
        assert(src_C == dst_C);
#ifdef DEBUG
        printf("random_crop: (%zu,%zu,%zu,%zu) -> (%zu,%zu,%zu,%zu)\n",
               src_N, src_C, src_H, src_W, dst_N, dst_C, dst_H, dst_W);
#endif
    }

	void reset(size_t src_N, size_t src_C, size_t src_H, size_t src_W,
                  size_t dst_N, size_t dst_C, size_t dst_H, size_t dst_W) {
		this->src_N = src_N;
		this->src_C = src_C;
		this->src_H = src_H;
		this->src_W = src_W;
		this->dst_N = dst_N;
		this->dst_C = dst_C;
		this->dst_H = dst_H;
		this->dst_W = dst_W;
	}

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};

template<class value_type>
class central_crop_t : public base_preprocess_t<value_type> {
private:
    size_t src_N, src_C, src_H, src_W, dst_N, dst_C, dst_H, dst_W;
public:
    central_crop_t(size_t src_N, size_t src_C, size_t src_H, size_t src_W,
                  size_t dst_N, size_t dst_C, size_t dst_H, size_t dst_W) :
        src_N(src_N), src_C(src_C), src_H(src_H), src_W(src_W),
        dst_N(dst_N), dst_C(dst_C), dst_H(dst_H), dst_W(dst_W) {
        assert(src_N == dst_N);
        assert(src_C == dst_C);
#ifdef DEBUG
        printf("central_crop: (%zu,%zu,%zu,%zu) -> (%zu,%zu,%zu,%zu)\n",
               src_N, src_C, src_H, src_W, dst_N, dst_C, dst_H, dst_W);
#endif
    }

	void reset(size_t src_N, size_t src_C, size_t src_H, size_t src_W,
			size_t dst_N, size_t dst_C, size_t dst_H, size_t dst_W) {
		this->src_N = src_N;
		this->src_C = src_C;
		this->src_H = src_H;
		this->src_W = src_W;
		this->dst_N = dst_N;
		this->dst_C = dst_C;
		this->dst_H = dst_H;
		this->dst_W = dst_W;
	}

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class random_flip_left_right_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W;
public:
    random_flip_left_right_t(size_t N, size_t C, size_t H, size_t W) : N(N), C(C), H(H), W(W) {}

    void transfer(value_type *src, value_type *dst) override;

	void reset(size_t N, size_t C, size_t H, size_t W) {
		this->N = N;
		this->C = C;
		this->H = H;
		this->W = W;
	}

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class random_brightness_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W;
    value_type max_delta;
    bool is_src_regularized;
public:
    random_brightness_t(size_t N, size_t C, size_t H, size_t W, value_type max_delta, bool is_src_regularized = false) :
            N(N), C(C), H(H), W(W), max_delta(max_delta), is_src_regularized(is_src_regularized) {
        if (is_src_regularized) {
            assert((max_delta >= -1.0) && (max_delta <= 1.0));
        }
    }

	void reset(size_t N, size_t C, size_t H, size_t W, value_type max_delta, bool is_src_regularized = false) {
		this->N = N;
		this->C = C;
		this->H = H;
		this->W = W;
		this->max_delta = max_delta;
		this->is_src_regularized = is_src_regularized;
	}

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;

};


template<class value_type>
class random_contrast_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W;
    value_type lower, upper;
public:
    random_contrast_t(size_t N, size_t C, size_t H, size_t W, value_type lower, value_type upper) :
            N(N), C(C), H(H), W(W), lower(lower), upper(upper) {
        assert(lower <= upper);
    }

	void reset(size_t N, size_t C, size_t H, size_t W, value_type lower, value_type upper) {
		this->N = N;
		this->C = C;
		this->H = H;
		this->W = W;
		this->lower = lower;
		this->upper = upper;
	}

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class per_image_standardization_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W;
public:
    per_image_standardization_t(size_t N, size_t C, size_t H, size_t W) :
            N(N), C(C), H(H), W(W) {
    }

	void reset(size_t N, size_t C, size_t H, size_t W) {
		this->N = N;
		this->C = C;
		this->H = H;
		this->W = W;
	}

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class mean_subtraction_t : public base_preprocess_t<value_type> {
private:
    size_t N, C, H, W;

    value_type scale;
    float *mean_value = NULL;
    value_type *channel_mean = NULL;
    bool is_channel_mean;

public:
    mean_subtraction_t(size_t N_, size_t C_, size_t H_, size_t W_,
                       const char *mean_file_path, value_type scale = 1.0);

    mean_subtraction_t(size_t N_, size_t C_, size_t H_, size_t W_, value_type* channel_mean, value_type scale = 1.0);

	void reset(size_t N_, size_t C_, size_t H_, size_t W_, const char *mean_file_path, value_type scale = 1.0);
	void reset(size_t N_, size_t C_, size_t H_, size_t W_, value_type* channel_mean, value_type scale = 1.0);

    void transfer(value_type *src, value_type *dst) override;

    size_t output_size() override;

    size_t input_size() override;
};


template<class value_type>
class preprocessor {
private:
    std::vector<base_preprocess_t<value_type> *> processors;
    std::vector<value_type *> tmps;
public:
    preprocessor() {}

    ~preprocessor() {
        for (size_t i = 0; i < tmps.size(); ++i) {
            cudaFree(tmps[i]);
        }
    }

    preprocessor<value_type> *add_preprocess(base_preprocess_t<value_type> *processor);

	void clear_processors();

    size_t input_size() {
        CHECK_NOTNULL(processors[0]);
        return processors[0]->input_size();
    }

    size_t output_size() {
        return processors.back()->output_size();
    }

    // pass through all preprocessors inplace
    void process(value_type *src, value_type* dst);
};


}

#endif //ATP_PREPROCESS_H
