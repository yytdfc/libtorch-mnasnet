#include <torch/data/datasets/base.h>

#include <torch/data/example.h>
#include <torch/types.h>

class ImgSet : public Dataset<ImgSet> {
  public:
    /// The mode in which the dataset is loaded.
    enum class Mode { kTrain, kTest };

    /// Loads the MNIST dataset from the `root` path.
    ///
    /// The supplied `root` path should contain the *content* of the unzipped
    /// MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
    explicit MNIST(const std::string& root, Mode mode = Mode::kTrain)
      :images_(read_images(root, mode == Mode::kTrain)),
      targets_(read_targets(root, mode == Mode::kTrain)) {};

    /// Returns the `Example` at the given `index`.
    Example<> get(size_t index) override{
      return {images_[index], targets_[index]};
    };

    /// Returns the size of the dataset.
    optional<size_t> size() const override{
      return images_.size(0);
    };

    /// Returns true if this is the training subset of MNIST.
    bool is_train() const noexcept{
      return images_.size(0) == kTrainSize;
    };

    /// Returns all images stacked into a single tensor.
    const Tensor& images() const{
      return images_;
    };

    /// Returns all targets stacked into a single tensor.
    const Tensor& targets() const{
      return targets_;
    };

  private:
    Tensor images_, targets_;
};
