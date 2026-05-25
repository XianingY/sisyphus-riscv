#ifndef BASIC_SET_H
#define BASIC_SET_H

#include <vector>
#include <iostream>
#include <optional>

namespace pres {

using AffineExpr = std::vector<int>;

class BasicSet {
  // [A I -b] [x 1]^T = 0
  std::vector<AffineExpr> tableau;
  std::vector<int> denom;
public:
  BasicSet() {}
  BasicSet(const std::vector<AffineExpr> &tableau): tableau(tableau) {}
  void dump(std::ostream &os);

  bool empty();

  // Combine constraints from two sets (intersection).
  // Both sets must have the same number of variables (columns).
  BasicSet intersect(const BasicSet &other) const;

  // Check if this set is a subset of other: this ∩ ¬other == ∅.
  // Only works for single-constraint "other" (used for direction checks).
  bool isSubsetOf(const BasicSet &other) const;

  // Number of variable columns (excluding the constant column).
  int numVars() const;

  // Add a single constraint row.
  void addConstraint(const AffineExpr &row);

  // Existentially eliminate variables with Fourier-Motzkin projection.
  // Rows are affine inequalities row(x) >= 0.  The operation is exact over
  // rationals and is used as a conservative dependence-analysis accelerator:
  // an empty projected set proves the original set empty, while a non-empty
  // projection may still be conservatively treated as "may depend".
  std::optional<BasicSet> projectOut(int var, size_t maxRows = 4096) const;
  std::optional<BasicSet> projectOut(const std::vector<int> &vars,
                                     size_t maxRows = 4096) const;

  // Get the tableau for inspection.
  const std::vector<AffineExpr> &getTableau() const { return tableau; }
};

}

#endif
