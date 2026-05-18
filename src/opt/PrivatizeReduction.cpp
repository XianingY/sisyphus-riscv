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
    // Only consider i32 phis (not pointers or other types)
    if (op->getResultType() != Value::i32) continue;
    
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
    
    // Check if it's a pure reduction (only used in additive chains, not as address)
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
        
        if (isa<AddIOp>(user)) {
          if (S.count(user)) continue;
          if (S.count(user->getOperand(0).defining) && S.count(user->getOperand(1).defining)) {
            ok = false; break;
          }
          // Make sure this AddIOp is NOT used as a memory address.
          // If any use of this add is a LoadOp's address or StoreOp's address,
          // it's pointer arithmetic, not a reduction.
          for (auto addrUse : user->getUses()) {
            if (isa<sys::LoadOp>(addrUse) && addrUse->DEF(0) == user) {
              ok = false; break;
            }
            if (isa<sys::StoreOp>(addrUse) && addrUse->getOperandCount() >= 2 && addrUse->DEF(1) == user) {
              ok = false; break;
            }
            if (isa<AddLOp>(addrUse)) {
              ok = false; break; // used in pointer arithmetic
            }
          }
          if (!ok) break;
          S.insert(user);
          Q.push_back(user);
        } else if (isa<AddLOp>(user)) {
          // AddLOp is pointer arithmetic — NOT a valid reduction chain
          ok = false; break;
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

    // Reject if any value in the reduction chain is stored to memory
    // inside the loop. This means intermediate values are observable
    // and zeroing the initial value would corrupt them.
    for (auto sop : S) {
      for (auto user : sop->getUses()) {
        if (!L->contains(user->getParent()))
          continue;
        // Direct store of reduction value
        if (isa<sys::StoreOp>(user) && user->getOperandCount() >= 2 &&
            user->DEF(0) == sop) {
          ok = false;
          break;
        }
        // Used in address computation (AddLOp) → likely pointer, not reduction
        if (isa<AddLOp>(user)) {
          ok = false;
          break;
        }
        // Used in a multiply that feeds a store → also observable
        if (isa<MulIOp>(user) || isa<SubIOp>(user)) {
          for (auto muser : user->getUses()) {
            if (!L->contains(muser->getParent()))
              continue;
            if (isa<sys::StoreOp>(muser) && muser->getOperandCount() >= 2 &&
                muser->DEF(0) == user) {
              ok = false;
              break;
            }
          }
          if (!ok) break;
        }
      }
      if (!ok) break;
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
