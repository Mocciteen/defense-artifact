#ifndef TVM_RUNTIME_INPUT_ZERO_DITHER_H_
#define TVM_RUNTIME_INPUT_ZERO_DITHER_H_

#include <tvm/runtime/base.h>

namespace tvm {
namespace runtime {
namespace inputzerodither {

TVM_DLL bool IsEnabled();
TVM_DLL bool ShouldApply(const DLTensor* tensor);
TVM_DLL bool TryCopyFromBytes(const void* src_data, size_t nbytes, DLTensor* dst);
TVM_DLL bool TryCopyFromTensor(const DLTensor* src, DLTensor* dst);

}  // namespace inputzerodither
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_INPUT_ZERO_DITHER_H_
