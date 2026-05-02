// X86 late-MIR selected-store memory defense pass.
//
// This pass is the X86 late-MIR checker and fallback for selected memory writes.
// The primary implementation point is MemoryWritePatchIR.cpp.  MIR search scopes
// only limit where this pass checks or falls back; the defense object is always
// a concrete memory store.  Verification checks for the patch footprint near the
// selected store.  Fallback patching rewrites the source register just before
// the original store, so the final write target remains unchanged.

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "x86-cipher-memdef-selected-store"
#define PASS_NAME "x86-cipher-memdef-selected-store"
#define PASS_DESC "X86 selected scalar-f32 store memory defense"

namespace llvm {
FunctionPass *createX86CipherMemdefSelectedStorePass();
void initializeX86CipherMemdefSelectedStorePassPass(PassRegistry &);
} // namespace llvm

STATISTIC(NumSearchScopesVisited, "Number of code scopes searched");
STATISTIC(NumCandidateStores, "Number of scalar f32 stores identified");
STATISTIC(NumStoresVerified, "Number of scalar f32 stores with patch footprint");
STATISTIC(NumStoresMissingPatch,
          "Number of scalar f32 stores missing patch footprint");
STATISTIC(NumStoresRewritten, "Number of scalar f32 stores rewritten");

static cl::opt<bool>
    EnablePass(PASS_NAME, cl::Hidden, cl::init(false),
               cl::desc("Enable X86 selected-store memory defense"));

static cl::opt<bool> PatchStores(
    "x86-cipher-memdef-selected-store-patch", cl::Hidden, cl::init(false),
    cl::desc("Fallback: rewrite selected scalar f32 memory stores at MIR level"));

static cl::opt<bool> VerifyStores(
    "x86-cipher-memdef-selected-store-verify", cl::Hidden, cl::init(false),
    cl::desc("Verify selected MIR stores still carry the IR patch footprint"));

static cl::opt<bool> SearchAllScopes(
    "x86-cipher-memdef-selected-store-all-scopes", cl::Hidden,
    cl::init(false),
    cl::desc("Search all code scopes for candidate memory stores"));

static cl::list<std::string> SearchScopeNameTokens(
    "x86-cipher-memdef-selected-store-scope-token", cl::Hidden,
    cl::desc("Search code scopes whose name contains TOKEN"), cl::ZeroOrMore);

static cl::opt<bool>
    Verbose("x86-cipher-memdef-selected-store-verbose", cl::Hidden,
            cl::init(false),
            cl::desc("Print selected-store MIR diagnostics"));

namespace {

constexpr const char *kConfigSymbol = "__cipher_memdef_config";
constexpr int kHighMaskOffset = 16;
constexpr int kPayloadMaskOffset = 20;
constexpr int kFixedWindowBitsOffset = 24;
constexpr int kIncrementOffset = 28;
constexpr int kSignatureMaskOffset = 36;
constexpr int kGlobalSeedOffset = 40;

struct ScratchGPR {
  unsigned GPR64 = 0;
  unsigned GPR32 = 0;
  unsigned GPR8 = 0;
};

class X86CipherMemdefSelectedStorePass : public MachineFunctionPass {
public:
  static char ID;

  X86CipherMemdefSelectedStorePass() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return PASS_DESC; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const X86InstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;

  bool searchScopeAllowsStoreDiscovery(const MachineFunction &MF) const;
  bool isCandidateStore(const MachineInstr &MI, Register &SrcXMM) const;
  bool hasConfigReference(const MachineInstr &MI) const;
  bool isSameAddress(const X86AddressMode &A, const X86AddressMode &B) const;
  bool isLoadFromAddress(const MachineInstr &MI,
                         const X86AddressMode &Address) const;
  bool verifyPatchFootprint(const MachineFunction &MF,
                            const MachineInstr &StoreMI) const;
  bool sourceRegisterIsDeadAfter(const MachineInstr &MI,
                                 const MachineRegisterInfo &MRI,
                                 Register Reg) const;
  bool getScratchRegsBefore(const MachineInstr &BeforeMI,
                            const MachineRegisterInfo &MRI,
                            SmallVectorImpl<ScratchGPR> &Scratch,
                            bool &EFlagsAvailable) const;
  bool rewriteStore(MachineFunction &MF, MachineInstr &StoreMI) const;
};

} // namespace

char X86CipherMemdefSelectedStorePass::ID = 0;

INITIALIZE_PASS(X86CipherMemdefSelectedStorePass, PASS_NAME, PASS_DESC, false,
                false)

FunctionPass *llvm::createX86CipherMemdefSelectedStorePass() {
  return new X86CipherMemdefSelectedStorePass();
}

bool X86CipherMemdefSelectedStorePass::searchScopeAllowsStoreDiscovery(
    const MachineFunction &MF) const {
  if (SearchAllScopes) {
    return true;
  }

  const Function &F = MF.getFunction();
  if (F.hasFnAttribute("cipher-memdef") ||
      F.hasFnAttribute("cipher-memdef-mir")) {
    return true;
  }

  for (const std::string &Token : SearchScopeNameTokens) {
    if (!Token.empty() && MF.getName().contains(Token)) {
      return true;
    }
  }
  return false;
}

bool X86CipherMemdefSelectedStorePass::isCandidateStore(
    const MachineInstr &MI, Register &SrcXMM) const {
  switch (MI.getOpcode()) {
  case X86::MOVSSmr:
  case X86::VMOVSSmr:
    break;
  default:
    return false;
  }

  if (MI.getNumOperands() < 6 || !MI.getOperand(5).isReg()) {
    return false;
  }
  SrcXMM = MI.getOperand(5).getReg();
  return SrcXMM.isPhysical();
}

bool X86CipherMemdefSelectedStorePass::hasConfigReference(
    const MachineInstr &MI) const {
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isSymbol() && StringRef(MO.getSymbolName()) == kConfigSymbol) {
      return true;
    }
  }
  return false;
}

bool X86CipherMemdefSelectedStorePass::isSameAddress(
    const X86AddressMode &A, const X86AddressMode &B) const {
  if (A.BaseType != B.BaseType || A.Scale != B.Scale ||
      A.IndexReg != B.IndexReg || A.Disp != B.Disp || A.GV != B.GV) {
    return false;
  }
  if (A.BaseType == X86AddressMode::RegBase) {
    return A.Base.Reg == B.Base.Reg;
  }
  return A.Base.FrameIndex == B.Base.FrameIndex;
}

bool X86CipherMemdefSelectedStorePass::isLoadFromAddress(
    const MachineInstr &MI, const X86AddressMode &Address) const {
  switch (MI.getOpcode()) {
  case X86::MOV32rm:
  case X86::MOVSSrm:
  case X86::MOVSSrm_alt:
  case X86::VMOVSSrm:
  case X86::VMOVSSrm_alt:
  case X86::MOVDI2PDIrm:
  case X86::VMOVDI2PDIrm:
    break;
  default:
    return false;
  }
  if (MI.getNumOperands() < 6) {
    return false;
  }
  return isSameAddress(getAddressFromInstr(&MI, 1), Address);
}

bool X86CipherMemdefSelectedStorePass::verifyPatchFootprint(
    const MachineFunction &MF, const MachineInstr &StoreMI) const {
  const X86AddressMode StoreAddr = getAddressFromInstr(&StoreMI, 0);
  const MachineBasicBlock &MBB = *StoreMI.getParent();

  bool SawOldValueLoad = false;
  bool SawRuntimeConfig = false;
  unsigned Window = 96;
  auto It = StoreMI.getIterator();
  while (It != MBB.begin() && Window-- > 0) {
    --It;
    const MachineInstr &MI = *It;
    SawOldValueLoad |= isLoadFromAddress(MI, StoreAddr);
    SawRuntimeConfig |= hasConfigReference(MI);
    if (SawOldValueLoad && SawRuntimeConfig) {
      ++NumStoresVerified;
      if (Verbose) {
        errs() << "[x86-cipher-memdef-selected-store] " << MF.getName()
               << ": verified patch footprint for store at ";
        StoreMI.print(errs());
        errs() << '\n';
      }
      return true;
    }
  }

  ++NumStoresMissingPatch;
  errs() << "[x86-cipher-memdef-selected-store] " << MF.getName()
         << ": selected store has no nearby patch footprint; ";
  StoreMI.print(errs());
  errs() << '\n';
  return false;
}

bool X86CipherMemdefSelectedStorePass::sourceRegisterIsDeadAfter(
    const MachineInstr &MI, const MachineRegisterInfo &MRI, Register Reg) const {
  const MachineBasicBlock &MBB = *MI.getParent();
  LivePhysRegs Live(*TRI);
  Live.addLiveOuts(MBB);

  for (auto It = MBB.rbegin(), End = MBB.rend(); It != End; ++It) {
    const MachineInstr &Cur = *It;
    if (&Cur == &MI) {
      return Live.available(MRI, Reg);
    }
    Live.stepBackward(Cur);
  }
  return false;
}

bool X86CipherMemdefSelectedStorePass::getScratchRegsBefore(
    const MachineInstr &BeforeMI, const MachineRegisterInfo &MRI,
    SmallVectorImpl<ScratchGPR> &Scratch, bool &EFlagsAvailable) const {
  static constexpr ScratchGPR Candidates[] = {
      {X86::RAX, X86::EAX, X86::AL},   {X86::RCX, X86::ECX, X86::CL},
      {X86::RDX, X86::EDX, X86::DL},   {X86::RSI, X86::ESI, X86::SIL},
      {X86::RDI, X86::EDI, X86::DIL},  {X86::R8, X86::R8D, X86::R8B},
      {X86::R9, X86::R9D, X86::R9B},   {X86::R10, X86::R10D, X86::R10B},
      {X86::R11, X86::R11D, X86::R11B},
  };

  const MachineBasicBlock &MBB = *BeforeMI.getParent();
  LivePhysRegs Live(*TRI);
  Live.addLiveOuts(MBB);

  bool Found = false;
  for (auto It = MBB.rbegin(), End = MBB.rend(); It != End; ++It) {
    const MachineInstr &Cur = *It;
    if (&Cur == &BeforeMI) {
      Live.stepBackward(Cur);
      EFlagsAvailable = Live.available(MRI, X86::EFLAGS);
      Found = true;
      break;
    }
    Live.stepBackward(Cur);
  }
  if (!Found) {
    return false;
  }

  for (const ScratchGPR &Candidate : Candidates) {
    if (!Live.available(MRI, Candidate.GPR64)) {
      continue;
    }
    Scratch.push_back(Candidate);
    if (Scratch.size() == 5) {
      return true;
    }
  }
  return false;
}

bool X86CipherMemdefSelectedStorePass::rewriteStore(
    MachineFunction &MF, MachineInstr &StoreMI) const {
  Register SrcXMM;
  if (!isCandidateStore(StoreMI, SrcXMM)) {
    return false;
  }
  if (!sourceRegisterIsDeadAfter(StoreMI, MF.getRegInfo(), SrcXMM)) {
    errs() << "[x86-cipher-memdef-selected-store] " << MF.getName()
           << ": skipping store because source XMM remains live after the"
              " store\n";
    return false;
  }

  SmallVector<ScratchGPR, 5> Scratch;
  bool EFlagsAvailable = false;
  if (!getScratchRegsBefore(StoreMI, MF.getRegInfo(), Scratch,
                            EFlagsAvailable)) {
    errs() << "[x86-cipher-memdef-selected-store] " << MF.getName()
           << ": skipping store because fewer than five dead caller-saved GPRs"
              " are available\n";
    return false;
  }
  if (!EFlagsAvailable) {
    errs() << "[x86-cipher-memdef-selected-store] " << MF.getName()
           << ": skipping store because EFLAGS are live before the site\n";
    return false;
  }

  const X86AddressMode StoreAddr = getAddressFromInstr(&StoreMI, 0);
  const ScratchGPR &OldBits = Scratch[0];
  const ScratchGPR &BaseBits = Scratch[1];
  const ScratchGPR &PatchBits = Scratch[2];
  const ScratchGPR &UseOldMask = Scratch[3];
  const ScratchGPR &CfgAddr = Scratch[4];

  MachineBasicBlock &MBB = *StoreMI.getParent();
  const DebugLoc DL = StoreMI.getDebugLoc();

  BuildMI(MBB, StoreMI, DL, TII->get(X86::LEA64r), CfgAddr.GPR64)
      .addReg(X86::RIP)
      .addImm(1)
      .addReg(X86::NoRegister)
      .addExternalSymbol(kConfigSymbol)
      .addReg(X86::NoRegister);

  addFullAddress(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::MOV32rm), OldBits.GPR32),
      StoreAddr);

  BuildMI(MBB, StoreMI, DL, TII->get(X86::MOVSS2DIrr), BaseBits.GPR32)
      .addReg(SrcXMM);

  BuildMI(MBB, StoreMI, DL, TII->get(X86::MOV32rr), PatchBits.GPR32)
      .addReg(OldBits.GPR32);
  addRegOffset(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::AND32rm), PatchBits.GPR32)
          .addReg(PatchBits.GPR32),
      CfgAddr.GPR64, false, kSignatureMaskOffset);
  addRegOffset(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::CMP32rm))
          .addReg(PatchBits.GPR32),
      CfgAddr.GPR64, false, kFixedWindowBitsOffset);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::MOV32ri), UseOldMask.GPR32)
      .addImm(0);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::SETCCr), UseOldMask.GPR8)
      .addImm(X86::COND_E);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::NEG32r), UseOldMask.GPR32)
      .addReg(UseOldMask.GPR32);

  addFullAddress(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::LEA64r), PatchBits.GPR64),
      StoreAddr);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::MOV32rr), BaseBits.GPR32)
      .addReg(PatchBits.GPR32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::SHR64ri), PatchBits.GPR64)
      .addReg(PatchBits.GPR64)
      .addImm(32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::IMUL32rri), PatchBits.GPR32)
      .addReg(PatchBits.GPR32)
      .addImm(static_cast<int32_t>(0x9e3779b9U));
  BuildMI(MBB, StoreMI, DL, TII->get(X86::XOR32rr), BaseBits.GPR32)
      .addReg(BaseBits.GPR32)
      .addReg(PatchBits.GPR32);
  addRegOffset(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::XOR32rm), BaseBits.GPR32)
          .addReg(BaseBits.GPR32),
      CfgAddr.GPR64, false, kGlobalSeedOffset);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::MOV32rr), PatchBits.GPR32)
      .addReg(BaseBits.GPR32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::SHR32ri), PatchBits.GPR32)
      .addReg(PatchBits.GPR32)
      .addImm(16);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::XOR32rr), BaseBits.GPR32)
      .addReg(BaseBits.GPR32)
      .addReg(PatchBits.GPR32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::IMUL32rri), BaseBits.GPR32)
      .addReg(BaseBits.GPR32)
      .addImm(static_cast<int32_t>(0x7feb352dU));
  BuildMI(MBB, StoreMI, DL, TII->get(X86::MOV32rr), PatchBits.GPR32)
      .addReg(BaseBits.GPR32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::SHR32ri), PatchBits.GPR32)
      .addReg(PatchBits.GPR32)
      .addImm(15);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::XOR32rr), BaseBits.GPR32)
      .addReg(BaseBits.GPR32)
      .addReg(PatchBits.GPR32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::IMUL32rri), BaseBits.GPR32)
      .addReg(BaseBits.GPR32)
      .addImm(static_cast<int32_t>(0x846ca68bU));
  BuildMI(MBB, StoreMI, DL, TII->get(X86::MOV32rr), PatchBits.GPR32)
      .addReg(BaseBits.GPR32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::SHR32ri), PatchBits.GPR32)
      .addReg(PatchBits.GPR32)
      .addImm(16);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::XOR32rr), PatchBits.GPR32)
      .addReg(PatchBits.GPR32)
      .addReg(BaseBits.GPR32);
  addRegOffset(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::AND32rm), PatchBits.GPR32)
          .addReg(PatchBits.GPR32),
      CfgAddr.GPR64, false, kPayloadMaskOffset);
  addRegOffset(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::OR32rm), PatchBits.GPR32)
          .addReg(PatchBits.GPR32),
      CfgAddr.GPR64, false, kFixedWindowBitsOffset);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::AND32rr), OldBits.GPR32)
      .addReg(OldBits.GPR32)
      .addReg(UseOldMask.GPR32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::NOT32r), UseOldMask.GPR32)
      .addReg(UseOldMask.GPR32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::AND32rr), PatchBits.GPR32)
      .addReg(PatchBits.GPR32)
      .addReg(UseOldMask.GPR32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::OR32rr), OldBits.GPR32)
      .addReg(OldBits.GPR32)
      .addReg(PatchBits.GPR32);

  BuildMI(MBB, StoreMI, DL, TII->get(X86::MOV32rr), PatchBits.GPR32)
      .addReg(OldBits.GPR32);

  addRegOffset(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::AND32rm), PatchBits.GPR32)
          .addReg(PatchBits.GPR32),
      CfgAddr.GPR64, false, kPayloadMaskOffset);
  addRegOffset(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::ADD32rm), PatchBits.GPR32)
          .addReg(PatchBits.GPR32),
      CfgAddr.GPR64, false, kIncrementOffset);
  addRegOffset(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::AND32rm), PatchBits.GPR32)
          .addReg(PatchBits.GPR32),
      CfgAddr.GPR64, false, kPayloadMaskOffset);
  addRegOffset(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::OR32rm), PatchBits.GPR32)
          .addReg(PatchBits.GPR32),
      CfgAddr.GPR64, false, kFixedWindowBitsOffset);

  BuildMI(MBB, StoreMI, DL, TII->get(X86::MOVSS2DIrr), BaseBits.GPR32)
      .addReg(SrcXMM);
  addRegOffset(
      BuildMI(MBB, StoreMI, DL, TII->get(X86::AND32rm), BaseBits.GPR32)
          .addReg(BaseBits.GPR32),
      CfgAddr.GPR64, false, kHighMaskOffset);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::OR32rr), BaseBits.GPR32)
      .addReg(BaseBits.GPR32)
      .addReg(PatchBits.GPR32);
  BuildMI(MBB, StoreMI, DL, TII->get(X86::MOVDI2SSrr), SrcXMM)
      .addReg(BaseBits.GPR32);

  ++NumStoresRewritten;
  if (Verbose) {
    errs() << "[x86-cipher-memdef-selected-store] " << MF.getName()
           << ": rewrote scalar f32 store at ";
    StoreMI.print(errs());
    errs() << '\n';
  }
  return true;
}

bool X86CipherMemdefSelectedStorePass::runOnMachineFunction(
    MachineFunction &MF) {
  if ((!EnablePass && !VerifyStores && !PatchStores) ||
      skipFunction(MF.getFunction()) ||
      !searchScopeAllowsStoreDiscovery(MF)) {
    return false;
  }

  ++NumSearchScopesVisited;
  TII = MF.getSubtarget<X86Subtarget>().getInstrInfo();
  TRI = MF.getSubtarget<X86Subtarget>().getRegisterInfo();

  SmallVector<MachineInstr *, 16> Stores;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      Register SrcXMM;
      if (isCandidateStore(MI, SrcXMM)) {
        Stores.push_back(&MI);
      }
    }
  }

  NumCandidateStores += Stores.size();
  if (Verbose || ((EnablePass || VerifyStores) && Stores.empty())) {
    errs() << "[x86-cipher-memdef-selected-store] " << MF.getName()
           << ": scalar-f32-store candidates=" << Stores.size() << '\n';
  }

  if (VerifyStores) {
    unsigned Verified = 0;
    for (MachineInstr *StoreMI : Stores) {
      if (verifyPatchFootprint(MF, *StoreMI)) {
        ++Verified;
      }
    }
    if (Verbose || Verified != Stores.size()) {
      errs() << "[x86-cipher-memdef-selected-store] " << MF.getName()
             << ": verify=" << Verified << "/" << Stores.size() << '\n';
    }
  }

  if (!PatchStores) {
    return false;
  }

  bool Changed = false;
  for (MachineInstr *StoreMI : Stores) {
    Changed |= rewriteStore(MF, *StoreMI);
  }
  return Changed;
}
