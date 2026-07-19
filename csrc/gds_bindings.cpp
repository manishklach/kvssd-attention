#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>

namespace py = pybind11;

namespace {

std::string cufile_error(CUfileError_t status) {
  return "cuFile status=" + std::to_string(static_cast<int>(status.err)) +
         ", CUDA status=" + std::to_string(static_cast<int>(status.cu_err));
}

class GDSFile {
 public:
  explicit GDSFile(const std::string& path) {
    CUfileError_t driver_status = cuFileDriverOpen();
    TORCH_CHECK(driver_status.err == CU_FILE_SUCCESS,
                "cuFileDriverOpen failed: ", cufile_error(driver_status));
    driver_open_ = true;

    fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ < 0) {
      cuFileDriverClose();
      driver_open_ = false;
      TORCH_CHECK(false, "O_DIRECT open failed for ", path, ": ", std::strerror(errno));
    }

    CUfileDescr_t descriptor{};
    descriptor.handle.fd = fd_;
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t register_status = cuFileHandleRegister(&handle_, &descriptor);
    if (register_status.err != CU_FILE_SUCCESS) {
      ::close(fd_);
      fd_ = -1;
      cuFileDriverClose();
      driver_open_ = false;
      TORCH_CHECK(false, "cuFileHandleRegister failed: ", cufile_error(register_status));
    }
    registered_ = true;
  }

  GDSFile(const GDSFile&) = delete;
  GDSFile& operator=(const GDSFile&) = delete;

  ~GDSFile() { close(); }

  int64_t read(torch::Tensor destination, int64_t file_offset, int64_t size) {
    TORCH_CHECK(destination.is_cuda(), "destination must be CUDA");
    TORCH_CHECK(destination.scalar_type() == torch::kUInt8, "destination must be uint8");
    TORCH_CHECK(destination.is_contiguous(), "destination must be contiguous");
    TORCH_CHECK(file_offset >= 0 && size >= 0, "offset and size must be non-negative");
    TORCH_CHECK(destination.numel() >= size, "destination is smaller than requested size");
    c10::cuda::CUDAGuard guard(destination.device());
    const ssize_t result = cuFileRead(handle_, destination.data_ptr(), static_cast<size_t>(size),
                                      static_cast<off_t>(file_offset), 0);
    TORCH_CHECK(result >= 0, "cuFileRead failed: ", std::strerror(static_cast<int>(-result)),
                " (", result, ")");
    return static_cast<int64_t>(result);
  }

  void close() {
    if (registered_) {
      cuFileHandleDeregister(handle_);
      registered_ = false;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    if (driver_open_) {
      cuFileDriverClose();
      driver_open_ = false;
    }
  }

 private:
  int fd_ = -1;
  CUfileHandle_t handle_{};
  bool registered_ = false;
  bool driver_open_ = false;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<GDSFile>(module, "GDSFile")
      .def(py::init<const std::string&>())
      .def("read", &GDSFile::read)
      .def("close", &GDSFile::close);
}
