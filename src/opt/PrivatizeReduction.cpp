#include "LoopPasses.h"
#include <set>
#include <vector>

using namespace sys;

std::map<std::string, int> PrivatizeReduction::stats() {
  return {
    { "privatized", privatized }
  };
}

void PrivatizeReduction::runImpl(LoopInfo *L) {
  for (auto subloop : L->subloops) {
    runImpl(subloop);
  }

  auto header = L->header;
  if (!header) return;
  auto preheader = L->preheader;
  if (!preheader) return;

  Builder builder;

  for (auto op : header->getOps()) {
    if (!isa<PhiOp>(op)) continue;
    
    // Find V_init
    Value V_init = nullptr;
    int ext_count = 0;
    for (size_t i = 0; i < op->getOperandCount(); i++) {
      auto from = FROM(op->getAttrs()[i]);
      if (!L->contains(from)) {
        V_init = op->getOperand(i);
        ext_count++;
      }
    }
    
    if (ext_count != 1 || !V_init.defining) continue;
    
    // Check if it's a pure reduction
    std::set<Op*> S;
    std::vector<Op*> Q;
    S.insert(op);
    Q.push_back(op);
    
    bool ok = true;
    while (!Q.empty()) {
      auto curr = Q.back();
      Q.pop_back();
      
      for (auto user : curr->getUses()) {
        if (!L->contains(user->getParent())) {
          continue; // Escapes, will fix later
        }
        
        if (isa<AddIOp>(user) || isa<AddLOp>(user)) {
          if (S.count(user)) continue;
          if (S.count(user->getOperand(0).defining) && S.count(user->getOperand(1).defining)) {
            ok = false; break;
          }
          S.insert(user);
          Q.push_back(user);
        } else if (isa<PhiOp>(user)) {
          if (S.count(user)) continue;
          bool has_ext = false;
          for (size_t i = 0; i < user->getOperandCount(); i++) {
            auto from = FROM(user->getAttrs()[i]);
            if (!L->contains(from)) has_ext = true;
          }
          if (has_ext) {
            ok = false; break;
          }
          S.insert(user);
          Q.push_back(user);
        } else {
          ok = false; break;
        }
      }
      if (!ok) break;
    }
    
    if (!ok) continue;
    
    // Verify all Phis
    for (auto sop : S) {
      if (isa<PhiOp>(sop) && sop != op) {
        for (size_t i = 0; i < sop->getOperandCount(); i++) {
          if (!S.count(sop->getOperand(i).defining)) {
            ok = false; break;
          }
        }
      }
    }
    
    if (!ok) continue;
    
    // Transform
    privatized++;
    
    auto region = header->getParent();
    auto entry = region->getFirstBlock();
    builder.setToBlockStart(entry);
    Value zero;
    bool is_long = false;
    
    for (auto sop : S) {
      if (isa<AddLOp>(sop)) is_long = true;
    }
    
    zero = builder.create<IntOp>({new IntAttr(0)});
    
    for (size_t i = 0; i < op->getOperandCount(); i++) {
      auto from = FROM(op->getAttrs()[i]);
      if (!L->contains(from)) {
        op->setOperand(i, zero);
      }
    }
    
    // Fix outside uses
    for (auto sop : S) {
      std::vector<Op*> outside_uses;
      for (auto user : sop->getUses()) {
        if (!L->contains(user->getParent())) {
          outside_uses.push_back(user);
        }
      }
      
      for (auto user : outside_uses) {
        if (isa<PhiOp>(user)) {
          for (size_t i = 0; i < user->getOperandCount(); i++) {
            if (user->getOperand(i).defining == sop) {
              auto pred = FROM(user->getAttrs()[i]);
              builder.setBeforeOp(pred->getLastOp());
              Value add;
              if (is_long) add = builder.create<AddLOp>({sop, V_init});
              else add = builder.create<AddIOp>({sop, V_init});
              user->setOperand(i, add);
            }
          }
        } else {
          for (size_t i = 0; i < user->getOperandCount(); i++) {
            if (user->getOperand(i).defining == sop) {
              builder.setBeforeOp(user);
              Value add;
              if (is_long) add = builder.create<AddLOp>({sop, V_init});
              else add = builder.create<AddIOp>({sop, V_init});
              user->setOperand(i, add);
            }
          }
        }
      }
    }
  }
}

void PrivatizeReduction::run() {
  LoopAnalysis loop(module);
  loop.run();
  auto forests = loop.getResult();
  auto funcs = collectFuncs();
  
  for (auto func : funcs) {
    const auto &forest = forests[func];
    for (auto info : forest.getLoops()) {
      if (!info->getParent()) {
        runImpl(info);
      }
    }
  }
}
