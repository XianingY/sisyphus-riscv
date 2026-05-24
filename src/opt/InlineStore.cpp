#include "Passes.h"
#include <cstdlib>
#include <cstring>
#include <unordered_set>

using namespace sys;

std::map<std::string, int> InlineStore::stats() {
  return {
    { "inlined-stores", inlined },
    { "const-loads", constLoads }
  };
}

namespace {

bool hasStore(BasicBlock *bb) {
  for (auto x : bb->getOps()) {
    if (isa<StoreOp>(x))
      return true;
  }
  return false;
}

int getenvInt(const char *name, int fallback, int minValue, int maxValue) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  char *end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if ((end && *end) || value < minValue || value > maxValue)
    return fallback;
  return (int)value;
}

bool mayMentionGlobal(Op *addr, GlobalOp *glob) {
  if (!addr || !glob)
    return false;
  if (isa<GetGlobalOp>(addr))
    return NAME(addr) == NAME(glob);
  auto alias = addr->find<AliasAttr>();
  if (!alias)
    return false;
  if (alias->unknown)
    return true;
  return alias->location.count(glob);
}

bool constGlobalEscapes(GlobalOp *glob, ModuleOp *module) {
  for (auto call : module->findAll<CallOp>()) {
    for (auto operand : call->getOperands()) {
      auto def = operand.defining;
      if (def && def->getResultType() == Value::i64 && mayMentionGlobal(def, glob))
        return true;
    }
  }
  for (auto ret : module->findAll<ReturnOp>()) {
    for (auto operand : ret->getOperands()) {
      auto def = operand.defining;
      if (def && def->getResultType() == Value::i64 && mayMentionGlobal(def, glob))
        return true;
    }
  }
  return false;
}

bool hasStoreToConstGlobal(GlobalOp *glob, ModuleOp *module) {
  for (auto store : module->findAll<StoreOp>()) {
    if (store->getOperandCount() < 2)
      continue;
    auto addr = store->DEF(1);
    if (mayMentionGlobal(addr, glob))
      return true;
  }
  return false;
}

bool exactConstOffset(Op *addr, GlobalOp *glob, int &offset) {
  if (!addr || !glob)
    return false;
  if (isa<GetGlobalOp>(addr) && NAME(addr) == NAME(glob)) {
    offset = 0;
    return true;
  }
  auto alias = addr->find<AliasAttr>();
  if (!alias || alias->unknown || alias->location.size() != 1)
    return false;
  auto &[base, offsets] = *alias->location.begin();
  if (base != glob || offsets.size() != 1 || offsets[0] < 0)
    return false;
  offset = offsets[0];
  return true;
}

bool blockOrderBefore(Op *a, Op *b) {
  if (!a || !b || a->getParent() != b->getParent())
    return false;
  for (auto runner = a; runner; runner = runner->nextOp()) {
    if (runner == b)
      return true;
  }
  return false;
}

bool opDominatesUse(Op *def, Op *use) {
  if (!def || !use)
    return false;
  auto defBlock = def->getParent();
  auto useBlock = use->getParent();
  if (!defBlock || !useBlock)
    return false;
  if (defBlock == useBlock)
    return blockOrderBefore(def, use);
  return defBlock->dominates(useBlock);
}

bool sameConstValue(Op *a, Op *b) {
  if (!a || !b || a->opid != b->opid)
    return false;
  if (isa<IntOp>(a))
    return V(a) == V(b);
  if (isa<FloatOp>(a))
    return F(a) == F(b);
  return false;
}

Op *cloneConstValue(Builder &builder, Op *value) {
  if (isa<IntOp>(value))
    return builder.create<IntOp>({ new IntAttr(V(value)) });
  if (isa<FloatOp>(value))
    return builder.create<FloatOp>({ new FloatAttr(F(value)) });
  return nullptr;
}

bool inlineInitializedGlobalLoads(GlobalOp *glob, ModuleOp *module,
                                  int maxElems, Builder &builder,
                                  int &constLoads) {
  if (!glob || !glob->has<SizeAttr>())
    return false;
  int elemCount = SIZE(glob) / 4;
  if (elemCount <= 0 || elemCount > maxElems)
    return false;
  if (constGlobalEscapes(glob, module))
    return false;

  struct StoredConst {
    Op *store = nullptr;
    Op *value = nullptr;
  };

  std::vector<StoredConst> values(elemCount);
  bool sawStore = false;
  for (auto store : module->findAll<StoreOp>()) {
    if (store->getOperandCount() < 2)
      continue;
    auto addr = store->DEF(1);
    if (!mayMentionGlobal(addr, glob))
      continue;

    int offset = 0;
    if (!exactConstOffset(addr, glob, offset))
      return false;
    if (offset < 0 || offset % 4 != 0 || offset / 4 >= elemCount)
      return false;

    auto value = store->DEF(0);
    if (!isa<IntOp>(value) && !isa<FloatOp>(value))
      return false;

    int index = offset / 4;
    if (values[index].value && !sameConstValue(values[index].value, value))
      return false;
    values[index] = { store, value };
    sawStore = true;
  }
  if (!sawStore)
    return false;

  bool changed = false;
  auto loads = module->findAll<LoadOp>();
  for (auto load : loads) {
    if (load->getOperandCount() != 1 || load->getResultType() == Value::i64)
      continue;

    int offset = 0;
    if (!exactConstOffset(load->DEF(0), glob, offset))
      continue;
    if (offset < 0 || offset % 4 != 0 || offset / 4 >= elemCount)
      continue;

    auto known = values[offset / 4];
    if (!known.store || !known.value || known.value->getResultType() != load->getResultType())
      continue;
    if (!opDominatesUse(known.store, load))
      continue;

    builder.setBeforeOp(load);
    auto replacement = cloneConstValue(builder, known.value);
    if (!replacement)
      continue;
    load->replaceAllUsesWith(replacement);
    load->erase();
    constLoads++;
    changed = true;
  }

  return changed;
}

}

#define BAD { bad = true; break; }

void InlineStore::run() {
  auto gets = module->findAll<GetGlobalOp>();
  auto gMap = getGlobalMap();
  auto fMap = getFunctionMap();
  int maxConstElems =
      getenvInt("SISY_INLINE_CONST_ARRAY_MAX_ELEMS", 256, 1, 4096);

  Builder constBuilder;
  for (auto func : module->findAll<FuncOp>())
    func->getRegion()->updateDoms();

  for (auto &[_, glob] : gMap)
    inlineInitializedGlobalLoads(glob, module, maxConstElems, constBuilder, constLoads);

  for (auto &[name, glob] : gMap) {
    if (!glob->has<ConstAttr>())
      continue;
    bool fp = glob->has<FloatArrayAttr>();
    auto *iarr = fp ? nullptr : glob->find<IntArrayAttr>();
    auto *farr = fp ? glob->find<FloatArrayAttr>() : nullptr;
    int elemCount = fp ? (farr ? farr->size : 0) : (iarr ? iarr->size : 0);
    if (elemCount <= 0 || elemCount > maxConstElems)
      continue;
    if (constGlobalEscapes(glob, module) || hasStoreToConstGlobal(glob, module))
      continue;

    auto loads = module->findAll<LoadOp>();
    for (auto load : loads) {
      if (load->getOperandCount() != 1 || load->getResultType() == Value::i64)
        continue;
      int offset = 0;
      if (!exactConstOffset(load->DEF(0), glob, offset))
        continue;
      if (offset < 0 || offset % 4 != 0 || offset / 4 >= elemCount)
        continue;

      constBuilder.setBeforeOp(load);
      Op *replacement = nullptr;
      if (fp) {
        replacement = constBuilder.create<FloatOp>(
            { new FloatAttr(farr->vf[offset / 4]) });
      } else {
        replacement = constBuilder.create<IntOp>(
            { new IntAttr(iarr->vi ? iarr->vi[offset / 4] : 0) });
      }
      load->replaceAllUsesWith(replacement);
      load->erase();
      constLoads++;
    }
  }

  // For each global, records in which functions they're used.
  std::unordered_map<std::string, std::unordered_set<std::string>> used;
  for (auto get : gets)
    used[NAME(get)].insert(NAME(get->getParentOp()));

  // Remove unused globals, and find out ones only used in <once> functions.
  std::vector<std::string> queue;
  for (auto [k, v] : gMap) {
    if (used[k].empty() && !v->has<ImpureAttr>())
      v->erase();
    if (used[k].size() == 1) {
      auto name = *used[k].begin();
      if (fMap[name]->has<AtMostOnceAttr>())
        queue.push_back(k);
    }
  }

  for (auto [_, v] : fMap)
    v->getRegion()->updateDoms();

  for (auto gname : queue) {
    auto funcname = *used[gname].begin();
    Op *func = fMap[funcname];
    Builder builder;

    auto region = func->getRegion();
    auto entry = region->getFirstBlock();
    auto glob = gMap[gname];
    bool fp = glob->has<FloatArrayAttr>();
    bool bad = false;

    for (auto runner = entry; runner->succs.size();) {
      auto ops = runner->getOps();
      for (auto op : ops) {
        if (isa<LoadOp>(op)) {
          if (!op->DEF()->has<AliasAttr>())
            BAD

          auto alias = ALIAS(op->DEF());
          if (alias->location.size() > 1)
            BAD

          auto [base, offsets] = *alias->location.begin();
          if (offsets.size() > 1 || offsets[0] == -1)
            BAD
          if (base != glob)
            BAD

          auto offset = offsets[0];
          builder.setBeforeOp(op);
          if (fp) {
            auto attr = glob->get<FloatArrayAttr>();
            auto f = builder.create<FloatOp>({ new FloatAttr(attr->vf[offset / 4]) });
            op->replaceAllUsesWith(f);
          } else {
            auto attr = glob->get<IntArrayAttr>();
            auto i = builder.create<IntOp>({ new IntAttr(attr->vi[offset / 4]) });
            op->replaceAllUsesWith(i);
          }
          op->erase();
        }

        if (isa<StoreOp>(op)) {
          if (!op->DEF(1)->has<AliasAttr>())
            BAD

          auto alias = ALIAS(op->DEF(1));
          if (alias->location.size() > 1)
            BAD

          auto [base, offsets] = *alias->location.begin();
          if (offsets.size() > 1 || offsets[0] == -1)
            BAD
          if (base != glob)
            BAD

          auto offset = offsets[0];
          if (fp) {
            if (!isa<FloatOp>(op->DEF(0)))
              continue;

            auto attr = glob->get<FloatArrayAttr>();
            attr->vf[offset / 4] = F(op->DEF(0));
          } else {
            if (!isa<IntOp>(op->DEF(0)))
              continue;

            auto attr = glob->get<IntArrayAttr>();
            attr->vi[offset / 4] = V(op->DEF(0));
          }
          inlined++;
          op->erase();
        }
      }
      
      if (bad)
        break;

      auto term = runner->getLastOp();
      if (isa<GotoOp>(term) && TARGET(term)->preds.size() == 1)
        runner = TARGET(term);
      else if (isa<BranchOp>(term)) {
        // These globals behave as local variables.
        // So if all successors of one branch doesn't have any stores,
        // and it isn't a loop-back edge,
        // then it's alright to inline stores in another branch.
        auto ifso = TARGET(term);
        auto ifnot = ELSE(term);
        // Don't check too far (haven't thought of how to handle loops yet.)
        // Just check the next block.
        if (isa<ReturnOp>(ifnot->getLastOp()) && !hasStore(ifnot) && !ifso->dominates(runner))
          runner = ifso;
        else if (isa<ReturnOp>(ifso->getLastOp()) && !hasStore(ifso) && !ifnot->dominates(runner))
          runner = ifnot;
        else break;
      } else break;
    }

    // Update allzero attribute.
    if (fp) {
      auto attr = glob->get<FloatArrayAttr>();
      for (int i = 0; i < attr->size; i++) {
        if (attr->vf[i] != 0) {
          attr->allZero = false;
          break;
        }
      }
    } else {
      auto attr = glob->get<IntArrayAttr>();
      for (int i = 0; i < attr->size; i++) {
        if (attr->vi[i] != 0) {
          attr->allZero = false;
          break;
        }
      }
    }
  }
}
