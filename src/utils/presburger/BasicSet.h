#ifndef BASIC_SET_H
#define BASIC_SET_H

#include <vector>
#include <iostream>

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

  // Get the tableau for inspection.
  const std::vector<AffineExpr> &getTableau() const { return tableau; }
};

}

#endif
