#include "CleanupPasses.h"

#include <deque>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool defaultValue) {
  const char *value = std::getenv(name);
  if (!value)
    return defaultValue;
  return value[0] && std::strcmp(value, "0") != 0;
}

struct Lattice {
  enum Kind {
    Unknown,
    Constant,
    Overdefined,
  } kind = Unknown;
  int value = 0;
};

Lattice constant(int value) {
  return {Lattice::Constant, value};
}

Lattice overdefined() {
  return {Lattice::Overdefined, 0};
}

bool sameValue(const Lattice &a, const Lattice &b) {
  return a.kind == b.kind && (a.kind != Lattice::Constant || a.value == b.value);
}

Lattice mergeValue(const Lattice &a, const Lattice &b) {
  if (a.kind == Lattice::Unknown)
    return b;
  if (b.kind == Lattice::Unknown)
    return a;
  if (a.kind == Lattice::Overdefined || b.kind == Lattice::Overdefined)
    return overdefined();
  return a.value == b.value ? a : overdefined();
}

bool checkedBinary(int lhs, int rhs, char op, int &out) {
  long long value = 0;
  switch (op) {
  case '+':
    value = (long long) lhs + rhs;
    break;
  case '-':
    value = (long long) lhs - rhs;
    break;
  case '*':
    value = (long long) lhs * rhs;
    break;
  case '/':
    if (rhs == 0)
      return false;
    if (lhs == std::numeric_limits<int>::min() && rhs == -1)
      return false;
    value = lhs / rhs;
    break;
  case '%':
    if (rhs == 0)
      return false;
    if (lhs == std::numeric_limits<int>::min() && rhs == -1)
      return false;
    value = lhs % rhs;
    break;
  case '&':
    value = lhs & rhs;
    break;
  case '|':
    value = lhs | rhs;
    break;
  case '^':
    value = lhs ^ rhs;
    break;
  default:
    return false;
  }
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max())
    return false;
  out = (int) value;
  return true;
}

void removePhiIncomingFrom(BasicBlock *bb, BasicBlock *from) {
  for (auto *phi : bb->getPhis()) {
    auto ops = phi->getOperands();
    std::vector<Attr*> attrs;
    for (auto *attr : phi->getAttrs())
      attrs.push_back(attr->clone());

    phi->removeAllOperands();
    phi->removeAllAttributes();

    for (size_t i = 0; i < ops.size() && i < attrs.size(); i++) {
      if (isa<FromAttr>(attrs[i]) && FROM(attrs[i]) == from)
        continue;
      if (!isa<FromAttr>(attrs[i]))
        continue;
      phi->pushOperand(ops[i]);
      phi->add<FromAttr>(FROM(attrs[i]));
    }

    for (auto *attr : attrs)
      delete attr;
  }
}

Op *phiIncoming(Op *phi, BasicBlock *pred) {
  if (!phi || !pred || !isa<PhiOp>(phi))
    return nullptr;
  const auto &ops = phi->getOperands();
  const auto &attrs = phi->getAttrs();
  size_t count = std::min(ops.size(), attrs.size());
  for (size_t i = 0; i < count; i++) {
    if (!isa<FromAttr>(attrs[i]))
      continue;
    if (FROM(attrs[i]) == pred)
      return ops[i].defining;
  }
  return nullptr;
}

Op *edgeValue(Op *op, BasicBlock *pred, BasicBlock *through) {
  std::unordered_set<Op*> seen;
  while (op && isa<PhiOp>(op) && op->getParent() == through && !seen.count(op)) {
    seen.insert(op);
    auto incoming = phiIncoming(op, pred);
    if (!incoming)
      return nullptr;
    op = incoming;
  }
  return op;
}

bool replaceableValueOp(Op *op) {
  if (!op || op->getResultType() != Value::i32)
    return false;
  if (isa<IntOp>(op) || isa<PhiOp>(op) || isa<GetArgOp>(op) ||
      isa<LoadOp>(op) || isa<CallOp>(op) || isa<AllocaOp>(op) ||
      isa<GlobalOp>(op) || isa<GetGlobalOp>(op))
    return false;
  if (isa<StoreOp>(op) || isa<ReturnOp>(op) || isa<BranchOp>(op) ||
      isa<GotoOp>(op) || op->has<ImpureAttr>())
    return false;
  return true;
}

class SCCPEngine {
  Region *region;
  std::unordered_map<Op*, Lattice> values;
  std::unordered_set<BasicBlock*> executableBlocks;
  std::set<std::pair<BasicBlock*, BasicBlock*>> executableEdges;
  std::deque<Op*> ssaWorklist;
  std::deque<BasicBlock*> cfgWorklist;
  Builder builder;

  Lattice getValue(Op *op) {
    if (!op)
      return overdefined();
    auto it = values.find(op);
    if (it != values.end())
      return it->second;
    return Lattice{};
  }

  void markValue(Op *op, Lattice next) {
    if (!op)
      return;
    Lattice old = getValue(op);
    Lattice merged = mergeValue(old, next);
    if (sameValue(old, merged))
      return;
    values[op] = merged;
    for (auto *use : op->getUses())
      ssaWorklist.push_back(use);
  }

  void markBlock(BasicBlock *bb) {
    if (!bb || bb->getParent() != region)
      return;
    if (!executableBlocks.insert(bb).second)
      return;
    cfgWorklist.push_back(bb);
  }

  void markEdge(BasicBlock *from, BasicBlock *to) {
    if (!from || !to || to->getParent() != region)
      return;
    if (!executableEdges.insert({from, to}).second)
      return;
    markBlock(to);
    for (auto *phi : to->getPhis())
      ssaWorklist.push_back(phi);
  }

  bool isExecutableEdge(BasicBlock *from, BasicBlock *to) const {
    return executableEdges.count({from, to});
  }

  std::optional<int> constantOnEdge(Op *op, BasicBlock *pred, BasicBlock *through) {
    op = edgeValue(op, pred, through);
    if (!op)
      return std::nullopt;
    Lattice v = getValue(op);
    if (v.kind == Lattice::Constant)
      return v.value;
    return std::nullopt;
  }

  Lattice evalPhi(Op *op) {
    Lattice result;
    bool sawIncoming = false;
    for (int i = 0; i < op->getOperandCount(); i++) {
      if (i >= (int) op->getAttrs().size() || !isa<FromAttr>(op->getAttrs()[i]))
        continue;
      BasicBlock *from = FROM(op->getAttrs()[i]);
      if (!isExecutableEdge(from, op->getParent()))
        continue;
      result = mergeValue(result, getValue(op->getOperand(i).defining));
      sawIncoming = true;
    }
    return sawIncoming ? result : Lattice{};
  }

  Lattice evalOp(Op *op) {
    if (!op || op->getResultType() == Value::unit)
      return Lattice{};
    if (isa<IntOp>(op))
      return constant(V(op));
    if (isa<PhiOp>(op))
      return evalPhi(op);
    if (isa<GetArgOp>(op) || isa<LoadOp>(op) || isa<CallOp>(op) ||
        isa<AllocaOp>(op) || isa<GetGlobalOp>(op) || isa<GlobalOp>(op) ||
        op->getResultType() != Value::i32)
      return overdefined();

    std::vector<Lattice> operands;
    operands.reserve(op->getOperandCount());
    bool anyUnknown = false;
    for (auto operand : op->getOperands()) {
      Lattice v = getValue(operand.defining);
      if (v.kind == Lattice::Overdefined)
        return overdefined();
      if (v.kind == Lattice::Unknown)
        anyUnknown = true;
      operands.push_back(v);
    }
    if (anyUnknown)
      return Lattice{};

    if (op->getOperandCount() == 1) {
      int x = operands[0].value;
      if (isa<MinusOp>(op)) {
        if (x == std::numeric_limits<int>::min())
          return overdefined();
        return constant(-x);
      }
      if (isa<NotOp>(op))
        return constant(x == 0 ? 1 : 0);
      if (isa<SetNotZeroOp>(op))
        return constant(x != 0 ? 1 : 0);
      return overdefined();
    }

    if (op->getOperandCount() == 2) {
      int lhs = operands[0].value;
      int rhs = operands[1].value;
      int out = 0;
      if (isa<AddIOp>(op) && checkedBinary(lhs, rhs, '+', out)) return constant(out);
      if (isa<SubIOp>(op) && checkedBinary(lhs, rhs, '-', out)) return constant(out);
      if (isa<MulIOp>(op) && checkedBinary(lhs, rhs, '*', out)) return constant(out);
      if (isa<DivIOp>(op) && checkedBinary(lhs, rhs, '/', out)) return constant(out);
      if (isa<ModIOp>(op) && checkedBinary(lhs, rhs, '%', out)) return constant(out);
      if (isa<AndIOp>(op) && checkedBinary(lhs, rhs, '&', out)) return constant(out);
      if (isa<OrIOp>(op) && checkedBinary(lhs, rhs, '|', out)) return constant(out);
      if (isa<XorIOp>(op) && checkedBinary(lhs, rhs, '^', out)) return constant(out);
      if (isa<EqOp>(op)) return constant(lhs == rhs ? 1 : 0);
      if (isa<NeOp>(op)) return constant(lhs != rhs ? 1 : 0);
      if (isa<LtOp>(op)) return constant(lhs < rhs ? 1 : 0);
      if (isa<LeOp>(op)) return constant(lhs <= rhs ? 1 : 0);
      return overdefined();
    }

    if (isa<SelectOp>(op) && op->getOperandCount() == 3) {
      int cond = operands[0].value;
      return cond != 0 ? operands[1] : operands[2];
    }

    return overdefined();
  }

  void visitTerminator(Op *term) {
    if (!term)
      return;
    BasicBlock *bb = term->getParent();
    if (auto *go = dyn_cast<GotoOp>(term)) {
      markEdge(bb, TARGET(go));
      return;
    }
    if (auto *br = dyn_cast<BranchOp>(term)) {
      Lattice cond = getValue(br->DEF(0));
      if (cond.kind == Lattice::Constant) {
        markEdge(bb, cond.value != 0 ? TARGET(br) : ELSE(br));
        return;
      }
      if (cond.kind == Lattice::Overdefined) {
        markEdge(bb, TARGET(br));
        markEdge(bb, ELSE(br));
      }
    }
  }

  void visitBlock(BasicBlock *bb) {
    for (auto *op : bb->getOps())
      ssaWorklist.push_back(op);
    visitTerminator(bb->getLastOp());
  }

public:
  explicit SCCPEngine(Region *region): region(region) {}

  void solve() {
    region->updatePreds();
    BasicBlock *entry = region->getFirstBlock();
    executableBlocks.insert(entry);
    cfgWorklist.push_back(entry);

    while (!cfgWorklist.empty() || !ssaWorklist.empty()) {
      while (!cfgWorklist.empty()) {
        BasicBlock *bb = cfgWorklist.front();
        cfgWorklist.pop_front();
        visitBlock(bb);
      }

      if (ssaWorklist.empty())
        continue;
      Op *op = ssaWorklist.front();
      ssaWorklist.pop_front();
      if (!op || !op->getParent() || !executableBlocks.count(op->getParent()))
        continue;
      if (isa<BranchOp>(op) || isa<GotoOp>(op)) {
        visitTerminator(op);
        continue;
      }
      markValue(op, evalOp(op));
    }
  }

  int executableBlockCount() const {
    return (int) executableBlocks.size();
  }

  int rewriteConstants() {
    int replaced = 0;
    for (auto *bb : region->getBlocks()) {
      if (!executableBlocks.count(bb))
        continue;
      auto ops = bb->getOps();
      for (auto *op : ops) {
        auto it = values.find(op);
        if (it == values.end() || it->second.kind != Lattice::Constant)
          continue;
        if (!replaceableValueOp(op))
          continue;
        builder.setBeforeOp(op);
        builder.replace<IntOp>(op, { new IntAttr(it->second.value) });
        replaced++;
      }
    }
    return replaced;
  }

  int rewriteBranches() {
    int folded = 0;
    for (auto *bb : region->getBlocks()) {
      if (!executableBlocks.count(bb) || bb->getOpCount() == 0)
        continue;
      auto *br = dyn_cast<BranchOp>(bb->getLastOp());
      if (!br)
        continue;
      Lattice cond = getValue(br->DEF(0));
      if (cond.kind != Lattice::Constant)
        continue;
      BasicBlock *taken = cond.value != 0 ? TARGET(br) : ELSE(br);
      BasicBlock *dead = cond.value != 0 ? ELSE(br) : TARGET(br);
      removePhiIncomingFrom(dead, bb);
      builder.replace<GotoOp>(br, { new TargetAttr(taken) });
      folded++;
    }
    if (folded)
      region->updatePreds();
    return folded;
  }

  int threadConstantEdges() {
    if (!envEnabled("SISY_ENABLE_SCCP_THREADING", false))
      return 0;

    int threaded = 0;
    region->updatePreds();
    for (auto *bb : region->getBlocks()) {
      if (!bb || !executableBlocks.count(bb))
        continue;
      auto *br = dyn_cast<BranchOp>(bb->getLastOp());
      if (!br)
        continue;
      bool phiOnly = true;
      for (auto *op : bb->getOps()) {
        if (op == br)
          break;
        if (!isa<PhiOp>(op)) {
          phiOnly = false;
          break;
        }
      }
      if (!phiOnly)
        continue;

      std::vector<BasicBlock*> preds(bb->preds.begin(), bb->preds.end());
      for (auto *pred : preds) {
        if (!pred || !isExecutableEdge(pred, bb))
          continue;
        auto cond = constantOnEdge(br->DEF(0), pred, bb);
        if (!cond)
          continue;
        BasicBlock *target = (*cond != 0) ? TARGET(br) : ELSE(br);
        if (!target || target == bb || target->preds.count(pred))
          continue;

        bool ok = true;
        std::vector<std::pair<Op*, Op*>> incoming;
        for (auto *phi : target->getPhis()) {
          Op *fromBB = phiIncoming(phi, bb);
          if (!fromBB) {
            ok = false;
            break;
          }
          Op *mapped = edgeValue(fromBB, pred, bb);
          if (!mapped) {
            ok = false;
            break;
          }
          if (mapped->getParent() == bb && !isa<PhiOp>(mapped)) {
            ok = false;
            break;
          }
          incoming.push_back({phi, mapped});
        }
        if (!ok)
          continue;

        auto *predTerm = pred->getLastOp();
        bool rewired = false;
        if (auto *go = dyn_cast<GotoOp>(predTerm)) {
          if (TARGET(go) == bb) {
            TARGET(go) = target;
            rewired = true;
          }
        } else if (auto *pbr = dyn_cast<BranchOp>(predTerm)) {
          if (TARGET(pbr) == bb) {
            TARGET(pbr) = target;
            rewired = true;
          }
          if (ELSE(pbr) == bb) {
            ELSE(pbr) = target;
            rewired = true;
          }
        }
        if (!rewired)
          continue;

        for (auto &[phi, value] : incoming) {
          phi->pushOperand(value);
          phi->add<FromAttr>(pred);
        }
        removePhiIncomingFrom(bb, pred);
        threaded++;
      }
    }
    if (threaded)
      region->updatePreds();
    return threaded;
  }
};

}  // namespace

std::map<std::string, int> SCCP::stats() {
  return {
    {"replaced-values", replacedValues},
    {"folded-branches", foldedBranches},
    {"threaded-edges", threadedEdges},
    {"executable-blocks", executableBlocks},
  };
}

void SCCP::run() {
  for (auto *func : collectFuncs()) {
    SCCPEngine engine(func->getRegion());
    engine.solve();
    executableBlocks += engine.executableBlockCount();
    foldedBranches += engine.rewriteBranches();
    threadedEdges += engine.threadConstantEdges();
    replacedValues += engine.rewriteConstants();
  }
}
