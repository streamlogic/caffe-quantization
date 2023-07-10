#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/iou_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define GTHRESH 0.33f
#define PTHRESH 0.08f

namespace caffe {

template <typename Dtype>
void IouAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void IouAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  // IouAccuracy is a scalar; 0 axes.
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
    << "bottom blobs must have same N";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
    << "bottom blobs must have same C";
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2))
    << "bottom blobs must have same H";
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3))
    << "bottom blobs must have same W";
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    // Per-class accuracy is a vector; 1 axes.
    vector<int> top_shape_per_class(1);
    top_shape_per_class[0] = bottom[0]->shape(1);
    top[1]->Reshape(top_shape_per_class);
    nums_buffer_.Reshape(top_shape_per_class);
  }
}

template <typename Dtype>
void IouAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int outer_num = bottom[0]->shape(0);
  const int inner_num = bottom[0]->shape(2) * bottom[0]->shape(3);
  const int num_labels = bottom[0]->shape(1);
  vector<int> expected(num_labels);
  vector<int> tp(num_labels,0);
  vector<int> fp(num_labels,0);
  vector<int> fn(num_labels,0);
  vector<int> tr(num_labels,0);
  if (top.size() > 1) {
    caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
    caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  }
  for (int i = 0; i < outer_num; ++i) {
    for (int j = 0; j < inner_num; ++j) {
      int nz = 0;
      int no = 0;
      for (int c = 0; c < num_labels; c++) {
	Dtype e = bottom_label[(i*num_labels + c)*inner_num+j];
	if (e>0.0f) {
	  nz++;
	  tr[c]++;
	}
	expected[c] = e > GTHRESH;
	if (expected[c]) no++;
      }
      if (nz == num_labels || no > 0) {
	for (int c = 0; c < num_labels; c++) {
	  int p = bottom_data[(i*num_labels + c)*inner_num+j] > PTHRESH;
	  if (expected[c] > 0) {
	    if (p > 0) {
	      tp[c]++;
	    } else
	      fn[c]++;
	  } else if (p > 0)
	    fp[c]++;
	}
      } // else it's don't care
    }
  }

  int valid = 0;
  Dtype accuracy = 0.0f;
  for (int c = 0; c < num_labels; c++) {
    if (tr[c]>0) {
      accuracy += (float)tp[c]/(float)(tp[c]+fp[c]+fn[c]);
      valid++;
    }
  }

  // LOG(INFO) << "IouAccuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / valid;
  if (top.size() > 1) {
    for (int i = 0; i < top[1]->count(); ++i) {
      top[1]->mutable_cpu_data()[i] =
	(tr[i]>0) ? (float)tp[i]/(float)(tp[i]+fp[i]+fn[i]) : 1.0f;
    }
  }
  // IouAccuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(IouAccuracyLayer);
REGISTER_LAYER_CLASS(IouAccuracy);

}  // namespace caffe
