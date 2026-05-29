#include "LoopPasses.h"
#include "AnalysisManager.h"
#include "../utils/Matcher.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <initializer_list>
#include <iostream>

using namespace sys;

// Vectorize rejection reasons are tracked via the pass `stats()` map so the
// pipeline runtime can observe how many candidate loops were vetoed by each
// cost-model gate. The stat keys also serve as the canonical names of those
// gates and double as documentation of the rejection logic.
namespace { struct VecStats {
  int vectorized = 0;
  int rejectMultipleLatches = 0;
  int rejectShape = 0;
  int rejectStop = 0;
  int rejectStep = 0;
  int rejectTripCount = 0;
  int rejectControl = 0;
  int rejectCall = 0;
  int rejectInductionPhi = 0;
  int rejectPointerPhi = 0;
  int rejectMemoryDep = 0;
  int rejectAddress = 0;
  int rejectStreams = 0;
  int rejectOther = 0;
  int scalableVectorized = 0;
  int scalableReductions = 0;
  int scalableMaskedTails = 0;
  int slpPacked = 0;
};
static VecStats g_vecStats;
}
#define VECDEBUG(msg) do { if (std::getenv("SISY_DEBUG_VECTORIZE")) std::cerr << "[vectorize] " << msg << "\n"; } while (0)
#define VECREJECT_KIND(kind, msg) do { ++g_vecStats.kind; VECDEBUG(msg); return; } while (0)
#define VECREJECT(msg) VECREJECT_KIND(rejectOther, msg)

namespace {

// Cost-model knobs are environment-tunable so different targets and benchmark
// suites can experiment with thresholds without recompiling. The defaults are
// chosen for fixed 4-lane 128-bit SIMD (RVV with vlen=128 / NEON), where two
// vector iterations (8 scalar iters) already amortize the setup overhead.
int vecGetenvInt(const char *name, int fallback, int minValue, int maxValue) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  char *end = nullptr;
  long v = std::strtol(raw, &end, 10);
  if (!end || *end != '\0' || v < minValue || v > maxValue)
    return fallback;
  return (int) v;
}

int vecMinFixedTripCount() {
  return vecGetenvInt("SISY_VEC_MIN_TRIP_COUNT", 8, 4, 1024);
}

int vecMaxLoadStreams() {
  return vecGetenvInt("SISY_VEC_MAX_LOAD_STREAMS", 4, 1, 32);
}

bool vecEnvEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
         std::strcmp(raw, "FALSE") != 0;
}

std::vector<Value> vecValues(std::initializer_list<Value> values) {
  return std::vector<Value>(values);
}

bool isConstInt(Op *op, int value) {
  return isa<IntOp>(op) && V(op) == value;
}

Op *stripSinglePhi(Op *op) {
  while (op && isa<PhiOp>(op) && op->getOperandCount() == 1)
    op = op->DEF();
  return op;
}

int fixedTripCount(LoopInfo *info) {
  auto lower = stripSinglePhi(info->start);
  auto upper = stripSinglePhi(info->stop);
  auto step = stripSinglePhi(info->step);
  if (!lower || !upper || !step ||
      !isa<IntOp>(lower) || !isa<IntOp>(upper) || !isa<IntOp>(step))
    return -1;

  int stepValue = V(step);
  if (stepValue <= 0)
    return -1;

  int span = V(upper) - V(lower);
  if (span <= 0 || span % stepValue != 0)
    return -1;
  return span / stepValue;
}

bool isScaleByFour(Op *op, Op *iv) {
  if (!isa<MulIOp>(op) && !isa<MulLOp>(op))
    return false;

  auto lhs = op->DEF(0);
  auto rhs = op->DEF(1);
  return (lhs == iv && isConstInt(rhs, 4)) || (rhs == iv && isConstInt(lhs, 4));
}

Value::Type vectorTypeForScalar(Value::Type ty) {
  if (ty == Value::i32)
    return Value::i128;
  if (ty == Value::f32)
    return Value::f128;
  return Value::unit;
}

Value::Type scalableVectorTypeForScalar(Value::Type ty) {
  if (ty == Value::i32)
    return Value::vscale_i32;
  if (ty == Value::f32)
    return Value::vscale_f32;
  return Value::unit;
}

struct AddressParts {
  Op *base = nullptr;
  int offset = 0;
};

struct OperandPack {
  Op *first = nullptr;
  AddressParts base;
  bool contiguous = false;
  bool splat = false;
};

bool splitConstAddress(Op *addr, AddressParts &parts) {
  if (isa<AllocaOp>(addr) || isa<GetGlobalOp>(addr)) {
    parts.base = addr;
    parts.offset = 0;
    return true;
  }

  if (!isa<AddLOp>(addr))
    return false;

  Op *basePart = nullptr;
  Op *offsetPart = nullptr;
  if (isa<IntOp>(addr->DEF(0))) {
    offsetPart = addr->DEF(0);
    basePart = addr->DEF(1);
  } else if (isa<IntOp>(addr->DEF(1))) {
    offsetPart = addr->DEF(1);
    basePart = addr->DEF(0);
  } else {
    return false;
  }

  if (!splitConstAddress(basePart, parts))
    return false;
  parts.offset += V(offsetPart);
  return true;
}

bool sameContiguousAddress(const AddressParts &base, Op *candidate, int lane) {
  AddressParts parts;
  return splitConstAddress(candidate, parts) &&
         parts.base == base.base &&
         parts.offset == base.offset + lane * 4;
}

bool matchOperandPack(OperandPack &pack, Op *operand, int lane, Value::Type scalarTy) {
  if (operand->getResultType() != scalarTy)
    return false;

  if (lane == 0) {
    pack.first = operand;
    pack.splat = true;
    if (isa<LoadOp>(operand))
      pack.contiguous = splitConstAddress(operand->DEF(0), pack.base);
    return true;
  }

  bool splat = pack.splat && operand == pack.first;
  bool contiguous = pack.contiguous &&
    isa<LoadOp>(operand) &&
    sameContiguousAddress(pack.base, operand->DEF(0), lane);

  pack.splat = splat;
  pack.contiguous = contiguous;
  return splat || contiguous;
}

bool scalarBinaryToVector(Builder &builder, Op *scalar, Op *lhs, Op *rhs, Op *&out) {
  switch (scalar->opid) {
  case AddIOp::id:
    out = builder.create<AddVOp>(vecValues({ lhs, rhs }));
    return true;
  case SubIOp::id:
    out = builder.create<SubVOp>(vecValues({ lhs, rhs }));
    return true;
  case MulIOp::id:
    out = builder.create<MulVOp>(vecValues({ lhs, rhs }));
    return true;
  case AddFOp::id:
    out = builder.create<AddFVOp>(vecValues({ lhs, rhs }));
    return true;
  case SubFOp::id:
    out = builder.create<SubFVOp>(vecValues({ lhs, rhs }));
    return true;
  case MulFOp::id:
    out = builder.create<MulFVOp>(vecValues({ lhs, rhs }));
    return true;
  default:
    return false;
  }
}

bool byteRangesOverlap(int aOffset, int aSize, int bOffset, int bSize) {
  return aOffset < bOffset + bSize && bOffset < aOffset + aSize;
}

Op *materializeOperandPack(Builder &builder, const OperandPack &pack, Value::Type vectorTy) {
  if (pack.contiguous) {
    assert(isa<LoadOp>(pack.first));
    return builder.create<LoadOp>(vectorTy, vecValues({ pack.first->DEF(0) }), { new SizeAttr(16) });
  }

  assert(pack.splat);
  if (vectorTy == Value::f128)
    return builder.create<BroadcastFOp>(vecValues({ pack.first }));
  return builder.create<BroadcastOp>(vecValues({ pack.first }));
}

bool trySLPStorePack(const std::vector<Op*> &ops,
                     const std::unordered_map<Op*, int> &order,
                     int firstStoreIndex,
                     std::unordered_set<Op*> &removed) {
  std::vector<Op*> stores;
  for (int i = firstStoreIndex; i < (int) ops.size() && stores.size() < 4; i++) {
    if (removed.count(ops[i]))
      continue;
    if (isa<StoreOp>(ops[i]))
      stores.push_back(ops[i]);
  }
  if (stores.size() != 4)
    return false;

  Op *firstValue = stores[0]->DEF(0);
  Value::Type scalarTy = firstValue->getResultType();
  Value::Type vectorTy = vectorTypeForScalar(scalarTy);
  if (vectorTy == Value::unit)
    return false;

  AddressParts dstBase;
  if (!splitConstAddress(stores[0]->DEF(1), dstBase))
    return false;

  bool isBinaryPack = isa<AddIOp>(firstValue) || isa<SubIOp>(firstValue) || isa<MulIOp>(firstValue) ||
                      isa<AddFOp>(firstValue) || isa<SubFOp>(firstValue) || isa<MulFOp>(firstValue);
  OperandPack lhsPack, rhsPack;
  int firstRelevant = order.at(stores[0]);
  int lastStore = order.at(stores[3]);
  auto noteRelevant = [&](Op *op) {
    if (order.count(op))
      firstRelevant = std::min(firstRelevant, order.at(op));
  };

  if (!isBinaryPack) {
    OperandPack valuePack;
    for (int lane = 0; lane < 4; lane++) {
      auto store = stores[lane];
      auto value = store->DEF(0);
      if (value->getResultType() != scalarTy ||
          !sameContiguousAddress(dstBase, store->DEF(1), lane) ||
          !matchOperandPack(valuePack, value, lane, scalarTy))
        return false;
      noteRelevant(value);
    }

    if ((!valuePack.contiguous && !valuePack.splat) ||
        (valuePack.splat && order.count(valuePack.first) && order.at(valuePack.first) >= order.at(stores[0])))
      return false;

    if (valuePack.contiguous && valuePack.base.base == dstBase.base &&
        valuePack.base.offset < dstBase.offset &&
        byteRangesOverlap(valuePack.base.offset, 16, dstBase.offset, 16))
      return false;

    std::unordered_set<Op*> packedStores(stores.begin(), stores.end());
    for (int i = firstRelevant; i <= lastStore; i++) {
      auto op = ops[i];
      if (isa<StoreOp>(op) && !packedStores.count(op))
        return false;
      if (isa<CallOp>(op) || isa<BranchOp>(op) || isa<GotoOp>(op) || isa<ReturnOp>(op))
        return false;
    }

    Builder builder;
    builder.setBeforeOp(stores[0]);
    auto vecValue = materializeOperandPack(builder, valuePack, vectorTy);
    builder.create<StoreOp>(vecValues({ vecValue, stores[0]->DEF(1) }), { new SizeAttr(16) });

    for (auto store : stores) {
      removed.insert(store);
      store->erase();
    }
    return true;
  }

  for (int lane = 0; lane < 4; lane++) {
    auto store = stores[lane];
    auto value = store->DEF(0);
    if (value->opid != firstValue->opid ||
        value->getResultType() != scalarTy ||
        !sameContiguousAddress(dstBase, store->DEF(1), lane))
      return false;

    auto lhs = value->DEF(0);
    auto rhs = value->DEF(1);
    if (!matchOperandPack(lhsPack, lhs, lane, scalarTy) ||
        !matchOperandPack(rhsPack, rhs, lane, scalarTy))
      return false;
    noteRelevant(lhs);
    noteRelevant(rhs);
  }

  if ((!lhsPack.contiguous && !lhsPack.splat) ||
      (!rhsPack.contiguous && !rhsPack.splat))
    return false;
  if (lhsPack.splat && order.count(lhsPack.first) && order.at(lhsPack.first) >= order.at(stores[0]))
    return false;
  if (rhsPack.splat && order.count(rhsPack.first) && order.at(rhsPack.first) >= order.at(stores[0]))
    return false;

  std::unordered_set<Op*> packedStores(stores.begin(), stores.end());
  for (int i = firstRelevant; i <= lastStore; i++) {
    auto op = ops[i];
    if (isa<StoreOp>(op) && !packedStores.count(op))
      return false;
    if (isa<CallOp>(op) || isa<BranchOp>(op) || isa<GotoOp>(op) || isa<ReturnOp>(op))
      return false;
  }

  Builder builder;
  builder.setBeforeOp(stores[0]);
  auto lhsVec = materializeOperandPack(builder, lhsPack, vectorTy);
  auto rhsVec = materializeOperandPack(builder, rhsPack, vectorTy);
  Op *vecValue = nullptr;
  if (!scalarBinaryToVector(builder, firstValue, lhsVec, rhsVec, vecValue))
    return false;
  builder.create<StoreOp>(vecValues({ vecValue, stores[0]->DEF(1) }), { new SizeAttr(16) });

  for (auto store : stores) {
    removed.insert(store);
    store->erase();
  }
  return true;
}

void runSLPOnBlock(BasicBlock *bb) {
  std::vector<Op*> ops(bb->getOps().begin(), bb->getOps().end());
  std::unordered_map<Op*, int> order;
  for (int i = 0; i < (int) ops.size(); i++)
    order[ops[i]] = i;

  std::unordered_set<Op*> removed;
  for (int i = 0; i < (int) ops.size(); i++) {
    if (removed.count(ops[i]) || !isa<StoreOp>(ops[i]))
      continue;
    if (trySLPStorePack(ops, order, i, removed))
      g_vecStats.slpPacked++;
  }
}

bool tryScalableRVVLoop(LoopInfo *info) {
  if (!vecEnvEnabled("SISY_ENABLE_RVV", true) ||
      !vecEnvEnabled("SISY_ENABLE_RVV_VLA", true) ||
      !vecEnvEnabled("SISY_TARGET_RISCV", false))
    return false;
  if (!info || info->latches.size() != 1)
    return false;

  auto header = info->header;
  auto latch = info->getLatch();
  if (!header || !latch || !isa<BranchOp>(latch->getLastOp()))
    return false;

  if (!info->preheader) {
    BasicBlock *preheader = nullptr;
    for (auto pred : header->preds) {
      if (info->latches.count(pred))
        continue;
      if (preheader)
        return false;
      preheader = pred;
    }
    info->preheader = preheader;
  }
  if (!info->preheader || !info->induction || !info->stop ||
      !info->step || !isConstInt(info->step, 1))
    return false;

  int fixedTrip = fixedTripCount(info);
  if (fixedTrip > 0 && fixedTrip < vecMinFixedTripCount())
    return false;

  auto phis = header->getPhis();
  for (auto phi : phis) {
    if (phi != info->induction)
      return false;
  }

  for (auto bb : info->getBlocks()) {
    auto term = bb->getLastOp();
    if (isa<BranchOp>(term) && bb != latch)
      return false;
    for (auto op : bb->getOps())
      if (isa<CallOp>(op))
        return false;
  }

  std::unordered_map<Op*, Op*> base;
  std::function<Op*(Op*)> findBaseLocal = [&](Op *op) -> Op* {
    if (!op)
      return nullptr;
    if (base.count(op))
      return base[op];
    if (isa<AddLOp>(op)) {
      auto lhs = findBaseLocal(op->DEF(0));
      if (lhs)
        return base[op] = lhs;
      return base[op] = findBaseLocal(op->DEF(1));
    }
    if (isa<AllocaOp>(op) || isa<GetGlobalOp>(op))
      return base[op] = op;
    if (isa<PhiOp>(op)) {
      const auto &operands = op->getOperands();
      if (operands.empty())
        return base[op] = nullptr;
      auto b0 = findBaseLocal(operands[0].defining);
      for (int i = 1; i < operands.size(); i++)
        if (findBaseLocal(operands[i].defining) != b0)
          return base[op] = nullptr;
      return base[op] = b0;
    }
    return base[op] = nullptr;
  };

  std::unordered_set<Op*> stored, loaded;
  std::vector<Op*> loads, stores, addrs;
  std::unordered_map<Op*, int> memOrder;
  int memOrdinal = 0;
  for (auto bb : info->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<StoreOp>(op)) {
        memOrder[op] = memOrdinal++;
        stored.insert(findBaseLocal(op->DEF(1)));
        addrs.push_back(op->DEF(1));
        stores.push_back(op);
      }
      if (isa<LoadOp>(op)) {
        memOrder[op] = memOrdinal++;
        loaded.insert(findBaseLocal(op->DEF(0)));
        addrs.push_back(op->DEF(0));
        loads.push_back(op);
      }
    }
  }

  if (loads.empty() || stores.empty() ||
      stored.count(nullptr) || loaded.count(nullptr))
    return false;
  if ((int) loads.size() > vecMaxLoadStreams())
    return false;

  for (auto load : loads) {
    auto loadBase = findBaseLocal(load->DEF(0));
    if (!stored.count(loadBase))
      continue;
    bool sameLaneUpdate = false;
    for (auto store : stores) {
      if (findBaseLocal(store->DEF(1)) != loadBase)
        continue;
      if (load->DEF(0) == store->DEF(1) &&
          load->getParent() == store->getParent() &&
          memOrder[load] < memOrder[store]) {
        sameLaneUpdate = true;
        break;
      }
    }
    if (!sameLaneUpdate)
      return false;
  }

  std::unordered_set<Op*> phiset(phis.begin(), phis.end());
  auto indexedBase = [&](Op *addr) -> Op* {
    if (!isa<AddLOp>(addr))
      return nullptr;
    auto lhs = addr->DEF(0);
    auto rhs = addr->DEF(1);
    if (isScaleByFour(rhs, info->induction))
      return findBaseLocal(lhs);
    if (isScaleByFour(lhs, info->induction))
      return findBaseLocal(rhs);
    return nullptr;
  };
  for (auto addr : addrs)
    if (!phiset.count(addr) && !indexedBase(addr))
      return false;

  Op *insertBefore = nullptr;
  for (auto op : header->getOps()) {
    if (!isa<PhiOp>(op)) {
      insertBefore = op;
      break;
    }
  }
  if (!insertBefore)
    return false;

  Builder builder;
  builder.setBeforeOp(insertBefore);
  auto avl = builder.create<SubIOp>(vecValues({ info->stop, info->induction }));
  auto vl = builder.create<VSetVLOp>(
      Value::i32, vecValues({ avl }), { new VectorShapeAttr(32) });

  std::vector<Op*> erased, created = { avl, vl };
  std::unordered_map<Op*, Op*> opmap;
  std::unordered_set<Op*> visited;
  Value::Type vectorTy = Value::unit;
  bool success = true;

  auto remember = [&](Op *old, Op *op) {
    opmap[old] = op;
    created.push_back(op);
    erased.push_back(old);
  };

  for (auto load : loads) {
    auto loadVectorTy = scalableVectorTypeForScalar(load->getResultType());
    if (loadVectorTy == Value::unit) {
      success = false;
      break;
    }
    if (vectorTy == Value::unit)
      vectorTy = loadVectorTy;
    if (vectorTy != loadVectorTy) {
      success = false;
      break;
    }
    visited.insert(load);
    builder.setBeforeOp(load);
    auto ld = builder.create<VScaleLoadOp>(
        loadVectorTy, vecValues({ load->DEF(0), vl }),
        { new VectorShapeAttr(32, true, false, false) });
    remember(load, ld);
  }

  std::deque<Op*> queue(stores.begin(), stores.end());
  while (success && !queue.empty()) {
    auto x = queue.back();
    queue.pop_back();
    if (visited.count(x))
      continue;

    bool ready = true;
    if (info->contains(x->getParent())) {
      const std::vector<Value> waitlist = isa<StoreOp>(x)
          ? std::vector<Value>{ x->getOperand(0) }
          : x->getOperands();
      for (auto operand : waitlist) {
        auto def = operand.defining;
        if (def && !visited.count(def) && info->contains(def->getParent()) &&
            !isa<PhiOp>(def)) {
          queue.push_back(def);
          ready = false;
        }
      }
      if (!ready) {
        queue.push_front(x);
        continue;
      }
    }

    visited.insert(x);
    builder.setBeforeOp(x);
    switch (x->opid) {
    case IntOp::id: {
      if (vectorTy != Value::vscale_i32) { success = false; break; }
      auto b = builder.create<VScaleBroadcastOp>(
          vectorTy, vecValues({ x, vl }), { new VectorShapeAttr(32) });
      created.push_back(b);
      opmap[x] = b;
      break;
    }
    case FloatOp::id: {
      if (vectorTy != Value::vscale_f32) { success = false; break; }
      auto b = builder.create<VScaleBroadcastOp>(
          vectorTy, vecValues({ x, vl }), { new VectorShapeAttr(32) });
      created.push_back(b);
      opmap[x] = b;
      break;
    }
    case LoadOp::id: {
      if (info->contains(x->getParent())) {
        success = false;
        break;
      }
      opmap[x] = x;
      break;
    }
    case StoreOp::id: {
      auto value = x->DEF(0);
      Op *vecValue = nullptr;
      if (opmap.count(value)) {
        vecValue = opmap[value];
      } else if ((isa<IntOp>(value) && vectorTy == Value::vscale_i32) ||
                 (isa<FloatOp>(value) && vectorTy == Value::vscale_f32) ||
                 !info->contains(value->getParent())) {
        vecValue = builder.create<VScaleBroadcastOp>(
            vectorTy, vecValues({ value, vl }), { new VectorShapeAttr(32) });
        created.push_back(vecValue);
      }
      if (!vecValue || vecValue->getResultType() != vectorTy) {
        success = false;
        break;
      }
      auto st = builder.create<VScaleStoreOp>(
          vecValues({ vecValue, x->DEF(1), vl }),
          { new VectorShapeAttr(32, true, false, false) });
      remember(x, st);
      break;
    }
    case AddIOp::id:
    case SubIOp::id:
    case MulIOp::id:
    case AddFOp::id:
    case SubFOp::id:
    case MulFOp::id: {
      auto a = x->DEF(0), b = x->DEF(1);
      if (opmap.count(b) && !opmap.count(a) &&
          (x->opid == AddIOp::id || x->opid == MulIOp::id ||
           x->opid == AddFOp::id || x->opid == MulFOp::id))
        std::swap(a, b);
      Op *va = opmap.count(a) ? opmap[a] : nullptr;
      Op *vb = opmap.count(b) ? opmap[b] : nullptr;
      if (!va && info->contains(a->getParent())) { success = false; break; }
      if (!vb && info->contains(b->getParent())) { success = false; break; }
      if (!va) {
        va = builder.create<VScaleBroadcastOp>(
            vectorTy, vecValues({ a, vl }), { new VectorShapeAttr(32) });
        created.push_back(va);
      }
      if (!vb) {
        vb = builder.create<VScaleBroadcastOp>(
            vectorTy, vecValues({ b, vl }), { new VectorShapeAttr(32) });
        created.push_back(vb);
      }
      Op *vop = nullptr;
      if (x->opid == AddIOp::id || x->opid == AddFOp::id)
        vop = builder.create<VScaleAddOp>(
            vectorTy, vecValues({ va, vb }), { new VectorShapeAttr(32) });
      else if (x->opid == SubIOp::id || x->opid == SubFOp::id)
        vop = builder.create<VScaleSubOp>(
            vectorTy, vecValues({ va, vb }), { new VectorShapeAttr(32) });
      else
        vop = builder.create<VScaleMulOp>(
            vectorTy, vecValues({ va, vb }), { new VectorShapeAttr(32) });
      remember(x, vop);
      break;
    }
    default:
      success = false;
      break;
    }

    if (success) {
      for (auto user : x->getUses())
        if (info->contains(user->getParent()))
          queue.push_back(user);
    }
  }

  if (!success || vectorTy == Value::unit) {
    for (auto op : created) {
      op->removeAllOperands();
      op->erase();
    }
    return false;
  }

  auto latchVal = Op::getPhiFrom(info->induction, latch);
  if (!latchVal || !isa<AddIOp>(latchVal)) {
    for (auto op : created) {
      op->removeAllOperands();
      op->erase();
    }
    return false;
  }
  latchVal->setOperand(1, vl);

  for (auto op : erased) {
    op->replaceAllUsesWith(opmap[op]);
    op->erase();
  }

  g_vecStats.vectorized++;
  g_vecStats.scalableVectorized++;
  g_vecStats.scalableMaskedTails++;
  if (std::getenv("SISY_DUMP_VECTOR_PLAN")) {
    std::cerr << "[vector-plan] rvv-vla loop=" << bbmap[header]
              << " loads=" << loads.size()
              << " stores=" << stores.size()
              << " type=" << (vectorTy == Value::vscale_f32 ? "f32" : "i32")
              << " tail=dynamic-vl\n";
  }
  return true;
}

} // namespace

// We can't just use the `Base` pass,
// as it doesn't handle phis at all.
Op *Vectorize::findBase(Op *op) {
  if (base.count(op))
    return base[op];

  if (isa<AddLOp>(op)) {
    auto lhs = findBase(op->DEF(0));
    if (lhs)
      return base[op] = lhs;
    return base[op] = findBase(op->DEF(1));
  }

  if (isa<AllocaOp>(op) || isa<GetGlobalOp>(op))
    return base[op] = op;

  if (isa<PhiOp>(op)) {
    const auto &operands = op->getOperands();
    base[op] = op;
    auto b0 = findBase(operands[0].defining);
    base[op] = b0;
    for (int i = 1; i < operands.size(); i++) {
      auto b = findBase(operands[i].defining);
      if (base[op] == op && b0 != op)
        base[op] = b0;
      
      if (b != base[op])
        return base[op] = nullptr;
    }
    return base[op] = b0;
  }

  // It doesn't have a base. (It might not even be a pointer.)
  return base[op] = nullptr;
}

void Vectorize::runImpl(LoopInfo *info) {
  base.clear();

  if (tryScalableRVVLoop(info))
    return;

  if (info->latches.size() > 1)
    VECREJECT_KIND(rejectMultipleLatches, "multiple latches");

  auto header = info->header;
  auto latch = info->getLatch();

  if (!info->preheader) {
    BasicBlock *preheader = nullptr;
    for (auto pred : header->preds) {
      if (info->latches.count(pred))
        continue;
      if (preheader)
        VECREJECT_KIND(rejectShape, "multiple external header predecessors");
      preheader = pred;
    }
    if (!preheader)
      VECREJECT_KIND(rejectShape, "missing external header predecessor");
    info->preheader = preheader;
  }

  // The loop must be rotated.
  if (!isa<BranchOp>(latch->getLastOp()))
    VECREJECT_KIND(rejectShape, "latch terminator is not branch");

  auto latchterm = latch->getLastOp();
  auto phis = header->getPhis();

  // Find an induction variable (even if it's addl).
  // The `LoopAnalysis` pass will only identify `addi`-formed induction variables.
  if (!info->getInduction()) {
    Rule brRotatedL("(br (lt (addl x z) y))");
    Rule brRotatedI("(br (lt (add x z) y))");
    Rule brI("(br (lt x y))");
    Rule addl("(addl x y)");
    Rule addi("(add x y)");

    // Try-bind one phi as the induction. Returns true on success.
    // `unitStepOnly` restricts the match to canonical `addi x 1` integer IVs;
    // when false, falls back to any addi/addl with a valid step.
    auto tryBind = [&](Op *phi, bool unitStepOnly) -> bool {
      auto def1 = Op::getPhiFrom(phi, info->preheader);
      auto def2 = Op::getPhiFrom(phi, latch);

      bool matchedAddL = !unitStepOnly && addl.match(def2, { { "x", phi } });
      bool matchedAddI = !matchedAddL && addi.match(def2, { { "x", phi } });
      if (!matchedAddL && !matchedAddI)
        return false;

      auto step = matchedAddL ? addl.extract("y") : addi.extract("y");

      if (!isa<IntOp>(step) && !step->getParent()->dominates(info->preheader))
        return false;
      if (isa<LoadOp>(step))
        return false;
      if (unitStepOnly && !isConstInt(step, 1))
        return false;

      auto term = latch->getLastOp();
      Op *stop = nullptr;
      if (brRotatedL.match(term, { { "x", phi } }))
        stop = brRotatedL.extract("y");
      else if (brRotatedI.match(term, { { "x", phi } }))
        stop = brRotatedI.extract("y");
      else if (brI.match(term, { { "x", def2 } }))
        stop = brI.extract("y");
      if (!stop)
        return false;

      info->induction = phi;
      info->start = def1;
      info->step = step;
      info->stop = stop;
      return true;
    };

    // Pass 1: prefer the canonical integer IV with const step 1. SCEV /
    // strength-reduction can introduce a stride-N pointer phi alongside the
    // original integer IV; without this preference, scanning order may pick
    // the pointer phi and downstream code rejects with "non-unit loop step".
    bool bound = false;
    for (auto phi : phis) {
      if (tryBind(phi, /*unitStepOnly=*/true)) { bound = true; break; }
    }

    // Pass 2: fall back to any addi/addl with a valid step (original logic).
    if (!bound) {
      for (auto phi : phis) {
        if (tryBind(phi, /*unitStepOnly=*/false)) break;
      }
    }
  }

  // Stop-fallback: LoopAnalysis only fills `stop` when the latch terminator
  // shape matches its narrow patterns. If induction was set but stop was not,
  // try to recover stop from the latch terminator here using the same rules
  // the bind path uses. Otherwise the matmul inner j-loop (and friends) get
  // rejected as "missing loop stop" even after rotation.
  if (info->induction && !info->stop) {
    Rule brRotatedL("(br (lt (addl x z) y))");
    Rule brRotatedI("(br (lt (add x z) y))");
    Rule brI("(br (lt x y))");
    auto phi = info->induction;
    auto def2 = Op::getPhiFrom(phi, latch);
    auto term = latch->getLastOp();
    if (brRotatedL.match(term, { { "x", phi } }))
      info->stop = brRotatedL.extract("y");
    else if (brRotatedI.match(term, { { "x", phi } }))
      info->stop = brRotatedI.extract("y");
    else if (def2 && brI.match(term, { { "x", def2 } }))
      info->stop = brI.extract("y");
  }

  // The loop must have a stop (in order to unroll below).
  if (!info->stop)
    VECREJECT_KIND(rejectStop, "missing loop stop");

  // This pass widens four i32 lanes at a time. Non-unit scalar IV steps need
  // a separate lane-address construction to stay correct.
  if (!info->step || !isConstInt(info->step, 1))
    VECREJECT_KIND(rejectStep, "non-unit loop step");

  // A fixed tiny loop does not amortize RVV/NEON setup and can be handled
  // better by scalar unrolling or straight scalar code. The threshold is
  // tunable via SISY_VEC_MIN_TRIP_COUNT to allow experimentation per workload.
  const int kMinFixedTripCount = vecMinFixedTripCount();
  int fixedTrip = fixedTripCount(info);
  if (fixedTrip > 0 && fixedTrip < kMinFixedTripCount)
    VECREJECT_KIND(rejectTripCount, "fixed trip count too small");

  // Ensure no branching except for the latch.
  for (auto bb : info->getBlocks()) {
    auto term = bb->getLastOp();
    if (isa<BranchOp>(term) && bb != latch) {
      VECREJECT_KIND(rejectControl, "branching block inside loop is not latch");
    }
  }

  // Ensure no calls anywhere.
  for (auto bb : info->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<CallOp>(op))
        VECREJECT_KIND(rejectCall, "call inside loop");
    }
  }

  std::unordered_set<Op*> bases;
  
  auto indexedBase = [&](Op *addr) -> Op* {
    if (!isa<AddLOp>(addr))
      return nullptr;

    auto lhs = addr->DEF(0);
    auto rhs = addr->DEF(1);
    if (isScaleByFour(rhs, info->induction))
      return findBase(lhs);
    if (isScaleByFour(lhs, info->induction))
      return findBase(rhs);
    return nullptr;
  };

  // Ensure all non-IV phis have pointer stride 4 and different bases.
  // The normal scalar-index form keeps the induction as addi + 1 and computes
  // addresses as base + iv * 4; those addresses are accepted below.
  for (auto phi : phis) {
    auto latchval = Op::getPhiFrom(phi, latch);

    if (phi == info->induction) {
      if (!isa<AddIOp>(latchval) || latchval->DEF(0) != phi || !isConstInt(latchval->DEF(1), 1))
        VECREJECT_KIND(rejectInductionPhi, "induction phi is not addi + 1");
      continue;
    }

    if (!isa<AddLOp>(latchval)) {
      // Other phi nodes cannot easily get mutated.
      VECREJECT_KIND(rejectPointerPhi, "unsupported non-induction phi");
    }

    auto base = findBase(latchval);
    if (!base || !isa<IntOp>(latchval->DEF(1)) || V(latchval->DEF(1)) != 4)
      VECREJECT_KIND(rejectPointerPhi, "pointer phi is not stride-4");

    if (bases.count(base))
      VECREJECT_KIND(rejectPointerPhi, "duplicate pointer phi base");
    bases.insert(base);
  }

  std::unordered_set<Op*> stored, loaded;
  std::vector<Op*> loads, stores, addrs;
  std::unordered_map<Op*, int> memOrder;
  int memOrdinal = 0;
  for (auto bb : info->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<StoreOp>(op)) {
        memOrder[op] = memOrdinal++;
        stored.insert(findBase(op->DEF(1))),
        addrs.push_back(op->DEF(1)),
        stores.push_back(op);
      }

      if (isa<LoadOp>(op)) {
        memOrder[op] = memOrdinal++;
        loaded.insert(findBase(op->DEF(0))),
        addrs.push_back(op->DEF(0)),
        loads.push_back(op);
      }
    }
  }

  // Accessed unknown places.
  if (stored.count(nullptr) || loaded.count(nullptr))
    VECREJECT_KIND(rejectAddress, "unknown memory base");

  // Keep the loop vectorizer from creating more live vector loads than the
  // current backends are prepared to schedule cheaply. The cap is tunable via
  // SISY_VEC_MAX_LOAD_STREAMS for backends with deeper vector register files.
  const int kMaxVectorLoadStreams = vecMaxLoadStreams();
  if ((int) loads.size() > kMaxVectorLoadStreams)
    VECREJECT_KIND(rejectStreams, "too many vector load streams");

  for (auto load : loads) {
    auto loadBase = findBase(load->DEF(0));
    if (!stored.count(loadBase))
      continue;

    bool sameLaneUpdate = false;
    for (auto store : stores) {
      if (findBase(store->DEF(1)) != loadBase)
        continue;
      if (load->DEF(0) == store->DEF(1) &&
          load->getParent() == store->getParent() &&
          memOrder[load] < memOrder[store]) {
        sameLaneUpdate = true;
        break;
      }
    }

    if (!sameLaneUpdate)
      VECREJECT_KIND(rejectMemoryDep, "loop-carried memory dependence");
  }

  // Ensure we only read/store to those phis.
  std::unordered_set<Op*> phiset(phis.begin(), phis.end());
  for (auto x : addrs) {
    if (!phiset.count(x) && !indexedBase(x))
      VECREJECT_KIND(rejectAddress, "memory address is not vectorizable");
  }

  // Start rewriting.
  std::vector<Op*> erased, created;
  bool success = true;

  std::deque<Op*> queue(stores.begin(), stores.end());
  
#define BAD(x) if (x) { VECDEBUG("bad on " #x); success = false; break; }

  Builder builder;
  std::unordered_map<Op*, Op*> opmap;
  std::unordered_set<Op*> visited;
  Value::Type vectorTy = Value::unit;

  const auto &replace = [&](Op *old, Op *op) {
    opmap[old] = op;
    created.push_back(op);
    erased.push_back(old);
  };

  // First deal with all loads.
  for (auto load : loads) {
    auto loadVectorTy = vectorTypeForScalar(load->getResultType());
    if (loadVectorTy == Value::unit) {
      success = false;
      break;
    }
    if (vectorTy == Value::unit)
      vectorTy = loadVectorTy;
    if (vectorTy != loadVectorTy) {
      success = false;
      break;
    }
    
    visited.insert(load);
    builder.setBeforeOp(load);

    auto ld = builder.create<LoadOp>(vectorTy, load->getOperands(), { new SizeAttr(16) });
    replace(load, ld);
    
    for (auto x : load->getUses())
      queue.push_back(x);
  }

  while (success && !queue.empty()) {
    // Pop from one end, and make sure the initial stores lie on the other end.
    // This will hopefully delay the stores after all their operands are prepared.
    auto x = queue.back();
    queue.pop_back();

    // If some of its operands inside the loop have not been visited, visit them first.
    bool ready = true;
    if (info->contains(x->getParent())) {
      const std::vector<Value> &waitlist = isa<StoreOp>(x)
        ? std::vector { x->getOperand(0) }
        : x->getOperands();
      
      for (auto operand : waitlist) {
        auto def = operand.defining;
        if (!visited.count(def) && info->contains(def->getParent()) && !isa<PhiOp>(def)) {
          queue.push_back(def);
          ready = false;
          continue;
        }
      }
      if (!ready) {
        queue.push_front(x);
        continue;
      }
    }

    if (visited.count(x))
      continue;
    visited.insert(x);

    builder.setBeforeOp(x);
    switch (x->opid) {
    case IntOp::id: {
      BAD(vectorTy != Value::i128);
      auto b = builder.create<BroadcastOp>(vecValues({ x }));
      created.push_back(b);
      break;
    }
    case FloatOp::id: {
      BAD(vectorTy != Value::f128);
      auto b = builder.create<BroadcastFOp>(vecValues({ x }));
      created.push_back(b);
      break;
    }
    case LoadOp::id: {
      // These loads are definitely outside the loop.
      // All loads inside this loop have been processed above.
      // Moreover, as all stores in this loop only stores to some of the phis,
      // the load must be loop-invariant.
      // Therefore just record it as it is.
      assert(!info->contains(x->getParent()));
      opmap[x] = x;
      break;
    }
    case StoreOp::id: {
      auto value = x->DEF(0);
      // Optimize memset-like operations.
      if (isa<IntOp>(value) || isa<FloatOp>(value) || !info->contains(value->getParent())) {
        auto b = vectorTy == Value::f128
          ? (Op*) builder.create<BroadcastFOp>(vecValues({ value }))
          : (Op*) builder.create<BroadcastOp>(vecValues({ value }));
        created.push_back(b);
        auto st = builder.create<StoreOp>(vecValues({ b, x->DEF(1) }), { new SizeAttr(16) });
        replace(x, st);
        break;
      }

      BAD(!opmap.count(value) || opmap[value]->getResultType() != vectorTy);
      auto st = builder.create<StoreOp>(vecValues({ opmap[value], x->DEF(1) }), { new SizeAttr(16) });
      replace(x, st);
      break;
    }
    case AddIOp::id: {
      auto a = x->DEF(0), b = x->DEF(1);
      if (opmap.count(a) && opmap.count(b)) {
        auto aty = opmap[a]->getResultType();
        auto bty = opmap[b]->getResultType();
        if (aty == bty && aty == Value::i128) {
          auto vop = builder.create<AddVOp>(vecValues({ opmap[a]->getResult(), opmap[b] }));
          replace(x, vop);
          break;
        }
      }
      if (opmap.count(b) && !opmap.count(a))
        std::swap(a, b);
      if (opmap.count(a) && !opmap.count(b)) {
        auto aty = opmap[a]->getResultType();
        auto bty = b->getResultType();
        
        // `b` has to be loop invariant.
        // TODO: when `b` itself is loop counter, perhaps optimizable?
        BAD(info->contains(b->getParent()));

        if (aty == Value::i128 && bty == Value::i32) {
          auto broadcast = builder.create<BroadcastOp>(vecValues({ b }));
          created.push_back(broadcast);
          auto viop = builder.create<AddVOp>(vecValues({ opmap[a]->getResult(), broadcast }));
          replace(x, viop);
          break;
        }
        if (aty == Value::i32 && bty == Value::i32) {
          auto add = builder.create<AddIOp>(vecValues({ opmap[a]->getResult(), b }));
          replace(x, add);
          break;
        }
      }
      success = false;
      break;
    }
    case SubIOp::id: {
      auto a = x->DEF(0), b = x->DEF(1);
      if (opmap.count(a) && opmap.count(b)) {
        auto aty = opmap[a]->getResultType();
        auto bty = opmap[b]->getResultType();
        if (aty == bty && aty == Value::i128) {
          auto vop = builder.create<SubVOp>(vecValues({ opmap[a]->getResult(), opmap[b] }));
          replace(x, vop);
          break;
        }
      }
      if (opmap.count(a) && !opmap.count(b)) {
        BAD(info->contains(b->getParent()));
        if (opmap[a]->getResultType() == Value::i128 && b->getResultType() == Value::i32) {
          auto broadcast = builder.create<BroadcastOp>(vecValues({ b }));
          created.push_back(broadcast);
          auto viop = builder.create<SubVOp>(vecValues({ opmap[a]->getResult(), broadcast }));
          replace(x, viop);
          break;
        }
      }
      if (!opmap.count(a) && opmap.count(b)) {
        BAD(info->contains(a->getParent()));
        if (a->getResultType() == Value::i32 && opmap[b]->getResultType() == Value::i128) {
          auto broadcast = builder.create<BroadcastOp>(vecValues({ a }));
          created.push_back(broadcast);
          auto viop = builder.create<SubVOp>(vecValues({ broadcast, opmap[b]->getResult() }));
          replace(x, viop);
          break;
        }
      }
      success = false;
      break;
    }
    case AddFOp::id: {
      auto a = x->DEF(0), b = x->DEF(1);
      if (opmap.count(a) && opmap.count(b)) {
        auto aty = opmap[a]->getResultType();
        auto bty = opmap[b]->getResultType();
        if (aty == bty && aty == Value::f128) {
          auto vop = builder.create<AddFVOp>(vecValues({ opmap[a]->getResult(), opmap[b] }));
          replace(x, vop);
          break;
        }
      }
      if (opmap.count(b) && !opmap.count(a))
        std::swap(a, b);
      if (opmap.count(a) && !opmap.count(b)) {
        auto aty = opmap[a]->getResultType();
        auto bty = b->getResultType();

        BAD(info->contains(b->getParent()));

        if (aty == Value::f128 && bty == Value::f32) {
          auto broadcast = builder.create<BroadcastFOp>(vecValues({ b }));
          created.push_back(broadcast);
          auto vfop = builder.create<AddFVOp>(vecValues({ opmap[a]->getResult(), broadcast }));
          replace(x, vfop);
          break;
        }
      }
      success = false;
      break;
    }
    case SubFOp::id: {
      auto a = x->DEF(0), b = x->DEF(1);
      if (opmap.count(a) && opmap.count(b)) {
        auto aty = opmap[a]->getResultType();
        auto bty = opmap[b]->getResultType();
        if (aty == bty && aty == Value::f128) {
          auto vop = builder.create<SubFVOp>(vecValues({ opmap[a]->getResult(), opmap[b] }));
          replace(x, vop);
          break;
        }
      }
      if (opmap.count(a) && !opmap.count(b)) {
        BAD(info->contains(b->getParent()));
        if (opmap[a]->getResultType() == Value::f128 && b->getResultType() == Value::f32) {
          auto broadcast = builder.create<BroadcastFOp>(vecValues({ b }));
          created.push_back(broadcast);
          auto vfop = builder.create<SubFVOp>(vecValues({ opmap[a]->getResult(), broadcast }));
          replace(x, vfop);
          break;
        }
      }
      if (!opmap.count(a) && opmap.count(b)) {
        BAD(info->contains(a->getParent()));
        if (a->getResultType() == Value::f32 && opmap[b]->getResultType() == Value::f128) {
          auto broadcast = builder.create<BroadcastFOp>(vecValues({ a }));
          created.push_back(broadcast);
          auto vfop = builder.create<SubFVOp>(vecValues({ broadcast, opmap[b]->getResult() }));
          replace(x, vfop);
          break;
        }
      }
      success = false;
      break;
    }
    case MulIOp::id: {
      auto a = x->DEF(0), b = x->DEF(1);
      if (opmap.count(a) && opmap.count(b)) {
        auto aty = opmap[a]->getResultType();
        auto bty = opmap[b]->getResultType();
        if (aty == bty && aty == Value::i128) {
          auto vop = builder.create<MulVOp>(vecValues({ opmap[a]->getResult(), opmap[b] }));
          replace(x, vop);
          break;
        }
      }
      if (opmap.count(b) && !opmap.count(a))
        std::swap(a, b);
      if (opmap.count(a) && !opmap.count(b)) {
        auto aty = opmap[a]->getResultType();
        auto bty = b->getResultType();
        
        // `b` has to be loop invariant.
        // TODO: when `b` itself is loop counter, perhaps optimizable?
        BAD(info->contains(b->getParent()));

        if (aty == Value::i128 && bty == Value::i32) {
          auto broadcast = builder.create<BroadcastOp>(vecValues({ b }));
          created.push_back(broadcast);
          auto viop = builder.create<MulVOp>(vecValues({ opmap[a]->getResult(), broadcast }));
          replace(x, viop);
          break;
        }
        if (aty == Value::i32 && bty == Value::i32) {
          auto add = builder.create<MulIOp>(vecValues({ opmap[a]->getResult(), b }));
          replace(x, add);
          break;
        }
      }
      success = false;
      break;
    }
    case MulFOp::id: {
      auto a = x->DEF(0), b = x->DEF(1);
      if (opmap.count(a) && opmap.count(b)) {
        auto aty = opmap[a]->getResultType();
        auto bty = opmap[b]->getResultType();
        if (aty == bty && aty == Value::f128) {
          auto vop = builder.create<MulFVOp>(vecValues({ opmap[a]->getResult(), opmap[b] }));
          replace(x, vop);
          break;
        }
      }
      if (opmap.count(b) && !opmap.count(a))
        std::swap(a, b);
      if (opmap.count(a) && !opmap.count(b)) {
        auto aty = opmap[a]->getResultType();
        auto bty = b->getResultType();

        BAD(info->contains(b->getParent()));

        if (aty == Value::f128 && bty == Value::f32) {
          auto broadcast = builder.create<BroadcastFOp>(vecValues({ b }));
          created.push_back(broadcast);
          auto vfop = builder.create<MulFVOp>(vecValues({ opmap[a]->getResult(), broadcast }));
          replace(x, vfop);
          break;
        }
      }
      success = false;
      break;
    }
    // TODO: check PhiOp and deal with accumulator?
    default:
      VECDEBUG("unhandled op " << x);
      success = false;
      break;
    }
  }

  if (!success) {
    ++g_vecStats.rejectOther;
    // Undo operations.
    VECDEBUG("undo for loop " << bbmap[info->header]);
    for (auto op : created)
      op->removeAllOperands();
    for (auto op : created)
      op->erase();
    return;
  }

  // Success.
  ++g_vecStats.vectorized;
  VECDEBUG("success, vectorized loop " << bbmap[info->header]);

  // Create a side loop.
  std::unordered_set<Op*> unwanted(created.begin(), created.end());
  auto exit = info->getExit();
  auto region = header->getParent();

  std::unordered_map<Op*, Op*> cloneMap;
  std::unordered_map<BasicBlock*, BasicBlock*> rewireMap;

  auto newpreheader = region->insert(exit);

  for (auto x : info->getBlocks())
    rewireMap[x] = region->insert(exit);

  // The new preheader should be connected to the new header.
  builder.setToBlockEnd(newpreheader);
  builder.create<GotoOp>({ new TargetAttr(rewireMap[header]) });

  // Shallow copy ops.
  for (auto [k, v] : rewireMap) {
    builder.setToBlockEnd(v);
    for (auto op : k->getOps()) {
      if (!unwanted.count(op)) {
        Op *cloned = builder.copy(op);
        cloneMap[op] = cloned;
      }
    }
  }

  // Rewire operands.
  for (auto &[old, cloned] : cloneMap) {
    for (int i = 0; i < old->getOperandCount(); i++) {
      if (cloneMap.count(old->DEF(i)))
        cloned->setOperand(i, cloneMap[old->DEF(i)]);
    }
  }

  // Rewire blocks.
  for (auto [_, v] : rewireMap) {
    auto term = v->getLastOp();
    if (auto attr = term->find<TargetAttr>(); attr && rewireMap.count(attr->bb))
      attr->bb = rewireMap[attr->bb];
    if (auto attr = term->find<ElseAttr>(); attr && rewireMap.count(attr->bb))
      attr->bb = rewireMap[attr->bb];
  }

  // The latch's exit branch should get to the new preheader instead.
  auto term = latch->getLastOp();
  if (TARGET(term) == exit)
    TARGET(term) = newpreheader;
  if (ELSE(term) == exit)
    ELSE(term) = newpreheader;

  auto tail = rewireMap[latch];

  // For the original loop, the stop should be decreased by 4 * step.
  // This is to guard against non-4-multiple things.
  auto cond = latchterm->DEF(0);
  auto preterm = info->preheader->getLastOp();
  builder.setBeforeOp(preterm);
  
  // Hoist the constants out of loop to fix dominance.
  auto stop = info->stop;
  if (isa<IntOp>(stop) && info->contains(stop->getParent()))
    stop = builder.create<IntOp>({ new IntAttr(V(stop)) });

  auto step = info->step;
  if (isa<IntOp>(step) && info->contains(step->getParent()))
    step = builder.create<IntOp>({ new IntAttr(V(step)) });
  
  Value four = builder.create<IntOp>({ new IntAttr(4) });
  Value mul = builder.create<MulIOp>(vecValues({ four, step }));
  Value lim = builder.create<SubLOp>(vecValues({ stop, mul }));
  builder.replace<LtOp>(cond, vecValues({ Op::getPhiFrom(info->induction, latch), lim }));

  // For the new header, everything comes from the new latch (`tail`).
  auto headerPhis = rewireMap[header]->getPhis();
  for (auto phi : headerPhis) {
    for (int i = 0; i < phi->getOperandCount(); i++) {
      auto attr = phi->getAttrs()[i];
      if (FROM(attr) == latch) {
        FROM(attr) = tail;
        break;
      }
    }
  }

  // For the exit, things can only come from the tail, and the values are also different.
  auto exitphis = exit->getPhis();
  for (auto phi : exitphis) {
    for (int i = 0; i < phi->getOperandCount(); i++) {
      auto attr = phi->getAttrs()[i];
      if (FROM(attr) == latch) {
        FROM(attr) = tail;
        phi->setOperand(i, cloneMap[phi->DEF(i)]);
        break;
      }
    }
  }

  // For the side loop, all values from preheader should come from the phis in the main loop.
  std::unordered_map<Op*, Op*> phiMap;
  for (auto phi : phis)
    phiMap[Op::getPhiFrom(phi, info->preheader)] = Op::getPhiFrom(phi, latch);
  
  for (auto phi : headerPhis) {
    for (int i = 0; i < phi->getOperandCount(); i++) {
      auto attr = phi->getAttrs()[i];
      if (FROM(attr) == info->preheader) {
        FROM(attr) = newpreheader;
        // Reset the operand to the value from the previous loop's latch.
        phi->setOperand(i, phiMap[phi->DEF(i)]);
        break;
      }
    }
  }

  // Commit operations and erase the original ones.
  for (auto op : erased) {
    op->replaceAllUsesWith(opmap[op]);
    op->erase();
  }

  // The stride is quadrapled.
  for (auto phi : phis) {
    auto latchval = Op::getPhiFrom(phi, latch);

    if (isa<AddIOp>(latchval) && isa<IntOp>(latchval->DEF(1))) {
      builder.setBeforeOp(latchval);
      auto more = builder.create<IntOp>({ new IntAttr(V(latchval->DEF(1)) * 4) });
      latchval->setOperand(1, more);
    }

    // We've ensured it's always 4 for every addl in question.
    if (isa<AddLOp>(latchval)) {
      builder.setBeforeOp(latchval);
      auto more = builder.create<IntOp>({ new IntAttr(16) });
      latchval->setOperand(1, more);
    }
  }
}

std::map<std::string, int> Vectorize::stats() {
  std::map<std::string, int> out;
  if (g_vecStats.vectorized)              out["vectorized-loops"]      = g_vecStats.vectorized;
  if (g_vecStats.rejectMultipleLatches)   out["reject-multi-latch"]    = g_vecStats.rejectMultipleLatches;
  if (g_vecStats.rejectShape)             out["reject-shape"]          = g_vecStats.rejectShape;
  if (g_vecStats.rejectStop)              out["reject-stop"]           = g_vecStats.rejectStop;
  if (g_vecStats.rejectStep)              out["reject-step"]           = g_vecStats.rejectStep;
  if (g_vecStats.rejectTripCount)         out["reject-trip-count"]     = g_vecStats.rejectTripCount;
  if (g_vecStats.rejectControl)           out["reject-control"]        = g_vecStats.rejectControl;
  if (g_vecStats.rejectCall)              out["reject-call"]           = g_vecStats.rejectCall;
  if (g_vecStats.rejectInductionPhi)      out["reject-induction-phi"]  = g_vecStats.rejectInductionPhi;
  if (g_vecStats.rejectPointerPhi)        out["reject-pointer-phi"]    = g_vecStats.rejectPointerPhi;
  if (g_vecStats.rejectMemoryDep)         out["reject-mem-dep"]        = g_vecStats.rejectMemoryDep;
  if (g_vecStats.rejectAddress)           out["reject-address"]        = g_vecStats.rejectAddress;
  if (g_vecStats.rejectStreams)           out["reject-streams"]        = g_vecStats.rejectStreams;
  if (g_vecStats.rejectOther)             out["reject-other"]          = g_vecStats.rejectOther;
  if (g_vecStats.scalableVectorized)      out["rvv-vla-vectorized"]    = g_vecStats.scalableVectorized;
  if (g_vecStats.scalableReductions)      out["rvv-vla-reductions"]    = g_vecStats.scalableReductions;
  if (g_vecStats.scalableMaskedTails)     out["rvv-vla-masked-tails"]  = g_vecStats.scalableMaskedTails;
  if (g_vecStats.slpPacked)               out["slp-packed-stores"]      = g_vecStats.slpPacked;
  return out;
}

void Vectorize::run() {
  g_vecStats = VecStats{};
  std::map<FuncOp*, LoopForest> localForests;
  std::map<FuncOp*, LoopForest> *forests = nullptr;
  if (context() && context()->enabled())
    forests = &context()->analysis().getLoopForests();
  else {
    LoopAnalysis analysis(module);
    analysis.run();
    localForests = analysis.getResult();
    forests = &localForests;
  }

  auto funcs = collectFuncs();
  
  for (auto func : funcs) {
    const auto &forest = (*forests)[func];

    // Only vectorize innermost loops.
    for (auto loop : forest.getLoops()) {
      if (!loop->subloops.size())
        runImpl(loop);
    }

    for (auto bb : func->getRegion()->getBlocks())
      runSLPOnBlock(bb);
  }
}

PreservedAnalyses Vectorize::run(PassContext &ctx) {
  activeContext = &ctx;
  run();
  activeContext = nullptr;
  bool changed = g_vecStats.vectorized || g_vecStats.slpPacked ||
                 g_vecStats.scalableVectorized;
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
