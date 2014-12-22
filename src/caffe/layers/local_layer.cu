#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/// @brief refer to CPU forward -- the BLAS implementation is the same.
template <typename Dtype>
void LocalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const int weight_offset = this->M_ * this->K_;
  const int col_offset = this->K_ * this->N_;
  const int top_offset = this->M_ * this->N_;
  
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    Dtype* col_data = this->col_buffer_.mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      // im2col transformation: unroll input regions for filtering
      // into column matrix for multplication.
      im2col_gpu(bottom_data + bottom[i]->offset(n), this->channels_, 
          this->height_, this->width_, this->kernel_h_, this->kernel_w_, 
          this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_,
          col_data);
      for (int g = 0; g < this->group_; ++g) {
        for (int m = 0 ;  m < this->num_output_; ++m)  {
          caffe_gpu_mul(this->K_ * this->N_, col_data + g * col_offset, 
              weight + this->blobs_[0]->offset(m) + g * weight_offset, 
              temp_.mutable_gpu_data());
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 1, this->N_,
              this->K_, (Dtype)1., all_ones_.gpu_data(), temp_.gpu_data(),
              (Dtype)0., top_data + (*top)[i]->offset(n,m) + g * top_offset);
        }
      }
      // Add bias.
      if (this->bias_term_) {
        if (this->layer_param_.convolution_param().shared_bias()) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->num_output_,
              this->N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
              this->bias_multiplier_.gpu_data(),
              (Dtype)1., top_data + (*top)[i]->offset(n));
        } else {
          caffe_gpu_add<Dtype>(this->blobs_[1]->count(),
              (*top)[i]->gpu_data() + (*top)[i]->offset(n),
              this->blobs_[1]->gpu_data(), top_data + (*top)[i]->offset(n));
        }
      }
    }
  }
}

/// @brief refer to CPU backward -- the BLAS implementation is the same.
template <typename Dtype>
void LocalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }
  const int weight_offset = this->M_ * this->K_;
  const int col_offset = this->K_ * this->N_;
  const int top_offset = this->M_ * this->N_;
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = NULL;
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      top_diff = top[i]->gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        caffe_gpu_gemv<Dtype>(CblasNoTrans, this->num_output_, this->N_,
            1., top_diff + top[0]->offset(n),
            this->bias_multiplier_.gpu_data(), 1.,
            bias_diff);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      if (!top_diff) {
        top_diff = top[i]->gpu_diff();
      }
      Dtype* col_data = this->col_buffer_.mutable_gpu_data();
      Dtype* col_diff = this->col_buffer_.mutable_gpu_diff();
      const Dtype* bottom_data = (*bottom)[i]->gpu_data();
      Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // Since we saved memory in the forward pass by not storing all col
        // data, we will need to recompute them.
        im2col_gpu(bottom_data + (*bottom)[i]->offset(n), this->channels_, 
            this->height_, this->width_, this->kernel_h_, this->kernel_w_, 
            this->pad_h_, this->pad_w_, this->stride_h_, this->stride_w_, 
            col_data);
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          for (int g = 0; g < this->group_; ++g) {
            for (int m = 0; m < this->num_output_; ++m) {
              for (int k = 0; k < this->K_; ++k) {
                caffe_gpu_mul(this->N_,
                    top_diff + top[i]->offset(n, m) + top_offset * g,
                    col_data + this->col_buffer_.offset(0, k) + col_offset * g,
                    temp_.mutable_gpu_data() + temp_.offset(0,  0, k));
              }
              caffe_gpu_axpby(this->K_ * this->N_, (Dtype)1., 
                  temp_.gpu_data(), (Dtype)1.,
                  weight_diff + this->blobs_[0]->offset(m) + g * weight_offset);
            }
          }
        }
        // gradient w.r.t. bottom data, if necessary
        if (propagate_down[i]) {
          if (weight == NULL) {
            weight = this->blobs_[0]->gpu_data();
          }
          for (int g = 0; g < this->group_; ++g) {
            for (int m = 0; m < this->num_output_; ++m) {
              for (int k = 0; k < this->K_; ++k) {
                caffe_gpu_mul(this->N_, top_diff + top[i]->offset(n, m) +
                    top_offset * g, 
                    weight + this->blobs_[0]->offset(m, 0, k),
                    temp2_.mutable_gpu_data());
                caffe_gpu_axpby(this->N_, (Dtype)1., temp2_.gpu_data(),
                    (Dtype)1.,
                    col_diff + this->col_buffer_.offset(0, k) + col_offset * g);
              }
            }
          }
          // col2im back to the data
          col2im_gpu(col_diff, this->channels_, this->height_, this->width_,
              this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_, 
              this->stride_h_, this->stride_w_,
              bottom_diff + (*bottom)[i]->offset(n));
        }
      }
    }
  }
}


INSTANTIATE_CLASS(LocalLayer);

}  // namespace caffe
