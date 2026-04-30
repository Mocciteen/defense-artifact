#ifndef TVM_RUNTIME_RELU_LOW12_PATCH_H_
#define TVM_RUNTIME_RELU_LOW12_PATCH_H_

#include <tvm/runtime/base.h>

namespace tvm {
namespace runtime {
namespace relulow12 {

bool IsEnabled();
void ResetInferencePatchSeed();
float PatchedReluFloat(float input, float old_output);
float PatchedRelu6FloatZeroOnly(float input, float old_output);

}  // namespace relulow12
}  // namespace runtime
}  // namespace tvm

extern "C" {

TVM_DLL float tvm_relu_low12_f32_scalar(float input, float old_output);
TVM_DLL float tvm_relu6_low12_f32_scalar(float input, float old_output);
TVM_DLL void tvm_relu_low12_reset_inference_seed(void);

}

#endif  // TVM_RUNTIME_RELU_LOW12_PATCH_H_
