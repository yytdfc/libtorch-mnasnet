#include <torch/data/datasets/base.h>

#include <torch/data/example.h>
#include <torch/types.h>
#include <opencv2/opencv.hpp>
#include <fstream>

namespace mnasnet{
  using namespace mnasnet;
  using namespace torch;
  using namespace torch::data;
class Cifar : public Dataset<Cifar> {
  public:
    /// The mode in which the dataset is loaded.
    enum class Mode { kTrain, kTest };

    /// Loads the MNIST dataset from the `root` path.
    ///
    /// The supplied `root` path should contain the *content* of the unzipped
    /// MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
    explicit Cifar(const std::string& root, Mode mode = Mode::kTrain) {
      ifstream fin(root + "/data_batch_1.bin", std::ios::binary);
      cout << root + "/data_batch_1.bin" << endl;
      uint8_t label;
      fin >> label;
      char img[3072];
      fin.read(img, 3072);
      cout << (int) img[1] << endl;
      Tensor t = torch::from_blob(img, {3, 32, 32}, torch::dtype(torch::kUInt8));
      cout << t.sizes() << " " << t[0][1][2] << " | " << t.flatten()[32 + 2] << " | " << (int) *(t.data<uint8_t>() + 1 * 32 + 2) << endl;
      t = t.permute({1, 2, 0});
      cout << t.sizes() << " " << t[0][1][2] << " | " << t.flatten()[3 + 2] << " | " << (int) *(t.data<uint8_t>() + 1 * 3 + 2) << endl;
      cout << t.sizes() << " " << t[0][1][2] << endl;
      // t.transpose_(0, 2);
      // auto a = cv::Mat(32, 32, CV_8UC3, t.permute({1, 2, 0}).data<uint8_t>());
      auto a = cv::Mat(32, 32, CV_8UC3, t.data<uint8_t>());
      cout << a.rows << " " << a.cols << endl;
      // auto a = cv::imread("/Users/fuchen/Downloads/111.jpeg");

      cv::imshow("00", a);

      cv::waitKey();

    };

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
      return true;
      // return mode==Mode::kTrain;
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
    int64_t kTrainSize = 20;
};
}
