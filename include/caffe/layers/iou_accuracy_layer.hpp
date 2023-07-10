#ifndef CAFFE_ACCURACY_LAYER_HPP_
#define CAFFE_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
template <typename Dtype>
class IouAccuracyLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides IouAccuracyParameter accuracy_param,
   *     with IouAccuracyLayer options:
   */
  explicit IouAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "IouAccuracy"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  // If there are two top blobs, then the second blob will contain
  // accuracies per class.
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlos() const { return 2; }

 protected:
  /**
   * @param bottom input Blob vector (length 2)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the predictions @f$ x @f$, a Blob with values in
   *      @f$ [0, 1] @f$ indicating the predicted score.
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the probabilities @f$ x @f$, a Blob with values in
   *      @f$ [0, 1] @f$ indicating the correct score.
   * @param top output Blob vector (length 1)
   *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
   *      the computed accuracy
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- IouAccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  int top_k_;

  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// Keeps counts of the number of samples per class.
  Blob<Dtype> nums_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
