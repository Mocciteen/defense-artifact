#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace {

enum class PatchKind {
  None,
  Generic,
  Relu,
  Relu6,
};

static cl::opt<bool> EnableAutoReluMatch(
    "cipher-memdef-auto-relu-match", cl::Hidden, cl::init(false),
    cl::desc("Patch stores whose value matches common ReLU/ReLU6 IR patterns"));

static cl::opt<bool> PatchAllFloatStores(
    "cipher-memdef-patch-all-float-stores", cl::Hidden, cl::init(false),
    cl::desc("Patch every supported float store in allowed search scopes"));

static cl::list<std::string> SearchScopeNameTokens(
    "cipher-memdef-scope-token", cl::Hidden,
    cl::desc("Restrict automatic store discovery to code scopes containing token"),
    cl::ZeroOrMore);

constexpr StringRef kConfigSymbol = "__cipher_memdef_config";

struct ConfigField {
  enum Index : unsigned {
    BitCount = 0,
    FixedBitCount = 1,
    PayloadBitCount = 2,
    LowMask = 3,
    HighMask = 4,
    PayloadMask = 5,
    FixedWindowBits = 6,
    Increment = 7,
    PatchPositive = 8,
    SignatureMask = 9,
    GlobalSeed = 10,
  };
};

bool isSupportedFloatStoreType(Type *type) {
  if (type->isFloatTy()) {
    return true;
  }
  auto *vector_type = dyn_cast<FixedVectorType>(type);
  return vector_type != nullptr && vector_type->getElementType()->isFloatTy();
}

Type *bitsTypeFor(Type *float_type, LLVMContext &context) {
  Type *i32 = Type::getInt32Ty(context);
  if (float_type->isFloatTy()) {
    return i32;
  }
  auto *vector_type = cast<FixedVectorType>(float_type);
  return FixedVectorType::get(i32, vector_type->getNumElements());
}

Constant *splatI32(Type *target_type, uint32_t value) {
  LLVMContext &context = target_type->getContext();
  Constant *scalar = ConstantInt::get(Type::getInt32Ty(context), value);
  if (!target_type->isVectorTy()) {
    return scalar;
  }
  auto *vector_type = cast<FixedVectorType>(target_type);
  return ConstantVector::getSplat(vector_type->getElementCount(), scalar);
}

Value *splatScalar(IRBuilder<> &builder, Value *scalar, Type *target_type) {
  if (!target_type->isVectorTy()) {
    return scalar;
  }
  auto *vector_type = cast<FixedVectorType>(target_type);
  Value *undef = UndefValue::get(target_type);
  Value *lane0 = builder.CreateInsertElement(undef, scalar, builder.getInt32(0));
  Constant *zero = builder.getInt32(0);
  Constant *mask = ConstantVector::getSplat(vector_type->getElementCount(), zero);
  return builder.CreateShuffleVector(lane0, undef, mask);
}

bool constantFloatEquals(const Constant *constant, double expected) {
  if (expected == 0.0 && constant->isNullValue()) {
    return true;
  }
  if (const auto *fp = dyn_cast<ConstantFP>(constant)) {
    return fp->isExactlyValue(expected);
  }
  if (const auto *data = dyn_cast<ConstantDataVector>(constant)) {
    for (unsigned i = 0, e = data->getNumElements(); i != e; ++i) {
      const auto *element = dyn_cast<ConstantFP>(data->getElementAsConstant(i));
      if (element == nullptr || !element->isExactlyValue(expected)) {
        return false;
      }
    }
    return true;
  }
  if (const auto *vector = dyn_cast<ConstantVector>(constant)) {
    for (unsigned i = 0, e = vector->getNumOperands(); i != e; ++i) {
      const auto *element = dyn_cast<ConstantFP>(vector->getOperand(i));
      if (element == nullptr || !element->isExactlyValue(expected)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool isConstantFloatLike(Value *value, double expected) {
  auto *constant = dyn_cast<Constant>(value);
  return constant != nullptr && constantFloatEquals(constant, expected);
}

bool isReluValue(Value *value);

bool isMinWithSix(Value *value, Value **inner) {
  if (auto *intrinsic = dyn_cast<IntrinsicInst>(value)) {
    if (intrinsic->getIntrinsicID() == Intrinsic::minnum &&
        intrinsic->arg_size() == 2) {
      Value *lhs = intrinsic->getArgOperand(0);
      Value *rhs = intrinsic->getArgOperand(1);
      if (isConstantFloatLike(lhs, 6.0)) {
        *inner = rhs;
        return true;
      }
      if (isConstantFloatLike(rhs, 6.0)) {
        *inner = lhs;
        return true;
      }
    }
  }

  auto *select = dyn_cast<SelectInst>(value);
  if (select == nullptr) {
    return false;
  }
  auto *cmp = dyn_cast<FCmpInst>(select->getCondition());
  if (cmp == nullptr) {
    return false;
  }

  Value *true_value = select->getTrueValue();
  Value *false_value = select->getFalseValue();
  Value *lhs = cmp->getOperand(0);
  Value *rhs = cmp->getOperand(1);

  if ((cmp->getPredicate() == CmpInst::FCMP_OLT ||
       cmp->getPredicate() == CmpInst::FCMP_OLE) &&
      true_value == lhs && false_value == rhs) {
    if (isConstantFloatLike(rhs, 6.0)) {
      *inner = lhs;
      return true;
    }
  }

  if ((cmp->getPredicate() == CmpInst::FCMP_OGT ||
       cmp->getPredicate() == CmpInst::FCMP_OGE) &&
      true_value == rhs && false_value == lhs) {
    if (isConstantFloatLike(rhs, 6.0)) {
      *inner = lhs;
      return true;
    }
  }

  return false;
}

bool isReluValue(Value *value) {
  if (auto *intrinsic = dyn_cast<IntrinsicInst>(value)) {
    if (intrinsic->getIntrinsicID() == Intrinsic::maxnum &&
        intrinsic->arg_size() == 2) {
      return isConstantFloatLike(intrinsic->getArgOperand(0), 0.0) ||
             isConstantFloatLike(intrinsic->getArgOperand(1), 0.0);
    }
  }

  auto *select = dyn_cast<SelectInst>(value);
  if (select == nullptr) {
    return false;
  }
  auto *cmp = dyn_cast<FCmpInst>(select->getCondition());
  if (cmp == nullptr) {
    return false;
  }

  Value *true_value = select->getTrueValue();
  Value *false_value = select->getFalseValue();
  Value *lhs = cmp->getOperand(0);
  Value *rhs = cmp->getOperand(1);

  if ((cmp->getPredicate() == CmpInst::FCMP_OGT ||
       cmp->getPredicate() == CmpInst::FCMP_OGE) &&
      true_value == lhs && false_value == rhs) {
    return isConstantFloatLike(rhs, 0.0);
  }

  if ((cmp->getPredicate() == CmpInst::FCMP_OLT ||
       cmp->getPredicate() == CmpInst::FCMP_OLE) &&
      true_value == rhs && false_value == lhs) {
    return isConstantFloatLike(rhs, 0.0);
  }

  return false;
}

PatchKind matchReluKind(Value *stored_value) {
  Value *inner = nullptr;
  if (isMinWithSix(stored_value, &inner) && isReluValue(inner)) {
    return PatchKind::Relu6;
  }
  if (isReluValue(stored_value)) {
    return PatchKind::Relu;
  }
  return PatchKind::None;
}

PatchKind metadataKind(const StoreInst &store) {
  MDNode *node = store.getMetadata("cipher.memdef.kind");
  if (node == nullptr || node->getNumOperands() == 0) {
    return PatchKind::None;
  }
  auto *text = dyn_cast<MDString>(node->getOperand(0).get());
  if (text == nullptr) {
    return PatchKind::None;
  }
  if (text->getString() == "patch" || text->getString() == "generic") {
    return PatchKind::Generic;
  }
  if (text->getString() == "relu") {
    return PatchKind::Relu;
  }
  if (text->getString() == "relu6") {
    return PatchKind::Relu6;
  }
  return PatchKind::None;
}

StringRef kindName(PatchKind kind) {
  switch (kind) {
  case PatchKind::Generic:
    return "generic";
  case PatchKind::Relu:
    return "relu";
  case PatchKind::Relu6:
    return "relu6";
  case PatchKind::None:
    return "none";
  }
  return "none";
}

bool searchScopeAllowsStoreDiscovery(const Function &function) {
  if (function.hasFnAttribute("cipher-memdef")) {
    return true;
  }
  if (SearchScopeNameTokens.empty()) {
    return true;
  }
  for (const std::string &token : SearchScopeNameTokens) {
    if (!token.empty() && function.getName().contains(token)) {
      return true;
    }
  }
  return false;
}

StructType *configType(LLVMContext &context) {
  Type *i32 = Type::getInt32Ty(context);
  return StructType::get(context,
                         {i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
                          i32});
}

GlobalVariable *configGlobal(Module &module) {
  if (auto *global = module.getGlobalVariable(kConfigSymbol)) {
    return global;
  }
  return new GlobalVariable(module, configType(module.getContext()), false,
                            GlobalValue::ExternalLinkage, nullptr,
                            kConfigSymbol);
}

Value *configFieldPointer(IRBuilder<> &builder, Module &module, unsigned index) {
  GlobalVariable *global = configGlobal(module);
  StructType *cfg_type = configType(module.getContext());
  return builder.CreateConstInBoundsGEP2_32(cfg_type, global, 0, index);
}

Value *loadConfigFieldScalar(IRBuilder<> &builder, Module &module,
                             unsigned index) {
  Value *ptr = configFieldPointer(builder, module, index);
  return builder.CreateLoad(builder.getInt32Ty(), ptr);
}

Value *loadConfigField(IRBuilder<> &builder, Module &module, unsigned index,
                       Type *target_type) {
  Value *ptr = configFieldPointer(builder, module, index);
  Value *scalar = builder.CreateLoad(builder.getInt32Ty(), ptr);
  return splatScalar(builder, scalar, target_type);
}

Value *mixSeed32(IRBuilder<> &builder, Value *value) {
  Value *mixed = builder.CreateXor(value, builder.CreateLShr(value, 16));
  mixed = builder.CreateMul(mixed, builder.getInt32(0x7feb352dU));
  mixed = builder.CreateXor(mixed, builder.CreateLShr(mixed, 15));
  mixed = builder.CreateMul(mixed, builder.getInt32(0x846ca68bU));
  return builder.CreateXor(mixed, builder.CreateLShr(mixed, 16));
}

Value *deriveAddressSeedScalar(IRBuilder<> &builder, Module &module,
                               Value *store_pointer, unsigned lane) {
  Value *address =
      builder.CreatePtrToInt(store_pointer, builder.getInt64Ty());
  if (lane != 0) {
    address = builder.CreateAdd(address, builder.getInt64(lane * 4ULL));
  }

  Value *low = builder.CreateTrunc(address, builder.getInt32Ty());
  Value *high =
      builder.CreateTrunc(builder.CreateLShr(address, 32), builder.getInt32Ty());
  Value *seed = loadConfigFieldScalar(builder, module, ConfigField::GlobalSeed);
  seed = builder.CreateXor(seed, low);
  seed = builder.CreateXor(
      seed, builder.CreateMul(high, builder.getInt32(0x9e3779b9U)));
  return mixSeed32(builder, seed);
}

Value *deriveAddressSeedBits(IRBuilder<> &builder, Module &module,
                             Value *store_pointer, Type *target_type,
                             Value *payload_mask, Value *fixed_window) {
  auto build_lane = [&](unsigned lane) {
    Value *seed = deriveAddressSeedScalar(builder, module, store_pointer, lane);
    Value *mask = target_type->isVectorTy()
                      ? loadConfigFieldScalar(builder, module,
                                              ConfigField::PayloadMask)
                      : payload_mask;
    Value *fixed = target_type->isVectorTy()
                       ? loadConfigFieldScalar(builder, module,
                                               ConfigField::FixedWindowBits)
                       : fixed_window;
    return builder.CreateOr(builder.CreateAnd(seed, mask), fixed);
  };

  if (!target_type->isVectorTy()) {
    return build_lane(0);
  }

  auto *vector_type = cast<FixedVectorType>(target_type);
  Value *result = UndefValue::get(target_type);
  for (unsigned i = 0, e = vector_type->getNumElements(); i != e; ++i) {
    result =
        builder.CreateInsertElement(result, build_lane(i), builder.getInt32(i));
  }
  return result;
}

Value *buildPatchedValue(IRBuilder<> &builder, Module &module, Value *base_value,
                         Value *old_value, Value *store_pointer,
                         PatchKind kind) {
  Type *float_type = base_value->getType();
  Type *bits_type = bitsTypeFor(float_type, module.getContext());

  Value *base_bits = builder.CreateBitCast(base_value, bits_type);
  Value *old_bits = builder.CreateBitCast(old_value, bits_type);

  Value *payload_mask =
      loadConfigField(builder, module, ConfigField::PayloadMask, bits_type);
  Value *fixed_window =
      loadConfigField(builder, module, ConfigField::FixedWindowBits, bits_type);
  Value *increment =
      loadConfigField(builder, module, ConfigField::Increment, bits_type);
  Value *high_mask =
      loadConfigField(builder, module, ConfigField::HighMask, bits_type);
  Value *signature_mask =
      loadConfigField(builder, module, ConfigField::SignatureMask, bits_type);

  Value *has_signature = builder.CreateICmpEQ(
      builder.CreateAnd(old_bits, signature_mask), fixed_window);
  Value *seed_bits =
      deriveAddressSeedBits(builder, module, store_pointer, bits_type,
                            payload_mask, fixed_window);
  Value *seed_source_bits =
      builder.CreateSelect(has_signature, old_bits, seed_bits);

  Value *payload = builder.CreateAnd(seed_source_bits, payload_mask);
  payload = builder.CreateAdd(payload, increment);
  payload = builder.CreateAnd(payload, payload_mask);
  Value *patched_low = builder.CreateOr(fixed_window, payload);
  Value *patched_bits =
      builder.CreateOr(builder.CreateAnd(base_bits, high_mask), patched_low);

  if (kind == PatchKind::Generic) {
    return builder.CreateBitCast(patched_bits, float_type);
  }

  Value *is_zero = builder.CreateICmpEQ(base_bits, splatI32(bits_type, 0));
  Value *exp_bits =
      builder.CreateAnd(base_bits, splatI32(bits_type, 0x7f800000u));
  Value *mantissa_bits =
      builder.CreateAnd(base_bits, splatI32(bits_type, 0x007fffffu));
  Value *is_nan = builder.CreateAnd(
      builder.CreateICmpEQ(exp_bits, splatI32(bits_type, 0x7f800000u)),
      builder.CreateICmpNE(mantissa_bits, splatI32(bits_type, 0)));

  Value *should_patch = is_zero;
  if (kind == PatchKind::Relu) {
    Value *patch_positive_scalar = loadConfigField(
        builder, module, ConfigField::PatchPositive, builder.getInt32Ty());
    Value *patch_positive =
        builder.CreateICmpNE(patch_positive_scalar, builder.getInt32(0));
    patch_positive =
        splatScalar(builder, patch_positive, should_patch->getType());
    should_patch = builder.CreateOr(should_patch, patch_positive);
  }
  should_patch = builder.CreateAnd(
      should_patch, builder.CreateNot(is_nan));

  Value *selected_bits =
      builder.CreateSelect(should_patch, patched_bits, base_bits);
  return builder.CreateBitCast(selected_bits, float_type);
}

class MemoryWritePatchIRPass : public PassInfoMixin<MemoryWritePatchIRPass> {
public:
  PreservedAnalyses run(Function &function, FunctionAnalysisManager &) {
    SmallVector<StoreInst *, 16> stores;
    for (Instruction &inst : instructions(function)) {
      if (auto *store = dyn_cast<StoreInst>(&inst)) {
        stores.push_back(store);
      }
    }

    bool changed = false;
    Module &module = *function.getParent();
    LLVMContext &context = module.getContext();
    const bool automatic_scope_allowed = searchScopeAllowsStoreDiscovery(function);

    for (StoreInst *store : stores) {
      if (store->isVolatile() || store->isAtomic()) {
        continue;
      }

      Value *stored_value = store->getValueOperand();
      if (!isSupportedFloatStoreType(stored_value->getType())) {
        continue;
      }

      PatchKind kind = metadataKind(*store);
      if (kind == PatchKind::None && automatic_scope_allowed &&
          PatchAllFloatStores) {
        kind = PatchKind::Generic;
      }
      if (kind == PatchKind::None && automatic_scope_allowed &&
          EnableAutoReluMatch) {
        kind = matchReluKind(stored_value);
      }
      if (kind == PatchKind::None) {
        continue;
      }

      IRBuilder<> builder(store);
      LoadInst *old_value = builder.CreateAlignedLoad(
          stored_value->getType(), store->getPointerOperand(), store->getAlign(),
          "cipher.memdef.old");
      Value *patched =
          buildPatchedValue(builder, module, stored_value, old_value,
                            store->getPointerOperand(), kind);
      store->setOperand(0, patched);

      Metadata *patched_metadata[] = {MDString::get(context, "ir"),
                                      MDString::get(context, kindName(kind))};
      MDNode *patched_node = MDNode::get(context, patched_metadata);
      store->setMetadata("cipher.memdef.patched", patched_node);
      changed = true;
    }

    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "cipher-memdef-store-patch", LLVM_VERSION_STRING,
          [](PassBuilder &pass_builder) {
            pass_builder.registerPipelineParsingCallback(
                [](StringRef name, FunctionPassManager &fpm,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name != "cipher-memdef-store-patch" &&
                      name != "cipher-memdef-relu-store") {
                    return false;
                  }
                  fpm.addPass(MemoryWritePatchIRPass());
                  return true;
                });
          }};
}
