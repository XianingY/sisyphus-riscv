#ifndef SISY_AST_TO_MLIR_H
#define SISY_AST_TO_MLIR_H

#include "SelfMLIR.h"

#include <memory>
#include <string>

namespace sys {
class ASTNode;
}

namespace sys::mlir {

std::unique_ptr<Module> lowerFromAST(Context &ctx, const sys::ASTNode &ast,
                                     const std::string &target,
                                     ProductionStats *stats = nullptr);

bool runProductionGateFromAST(const sys::ASTNode &ast, const std::string &target,
                              ProductionStats &stats, std::ostream *dump = nullptr);

} // namespace sys::mlir

#endif
