#include "BasicSet.h"
#include <numeric>
#include <sstream>
#include <iomanip>

using namespace pres;

// See this tutorial of dual simplex; I wrote it.
// https://gbc.xq.gl/polyhedral/maths/dual-simplex/
bool BasicSet::empty() {
  if (!denom.size())
    denom = std::vector<int>(tableau.size(), 1);

  for (;;) {
    // Choose a variable to evict from base.
    // We typically choose the x_r with minimal b_r.
    auto row = -1;
    auto min = 0;
    for (int i = 0; i < tableau.size(); ++i) {
      if (tableau[i].back() < min) {
        row = i;
        min = tableau[i].back();
      }
    }

    // Now every element in `b` is non-negative.
    // We've found a feasible solution.
    if (row == -1)
      return false;

    // We need to find a variable to put autoo the base, whose value has to be positive.
    // This means we must make sure A_{rj} < 0.
    // We typically choose the A_j with minimal c_j / A_{rj}; but we don't have a target `c` here.
    // Instead, we'll choose the first one encountered.
    auto col = -1;
    // Don't take `b_r` autoo account. That column (1) can't be pivoted.
    for (int j = 0; j < tableau[row].size() - 1; ++j) {
      if (tableau[row][j] < 0) {
        col = j;
        break;
      }
    }

    // No valid pivot. Infeasible.
    if (col == -1)
      return true;

    // Pivot.
    auto pivot = tableau[row][col];
    // Normalize the pivot row.
    // We would write:
    //   for (auto &x : tableau[row])
    //     x /= pivot;
    //
    // However, currently the pivot value is actually `pivot / denom[row]`,
    // so we'd multiply the denominator by `pivot` and everything else by denom[row].
    // (We've had gcd(tableau[row], pivot) == 1 after each pivot step, so no simplification possible here.)
    for (auto &x : tableau[row])
      x *= denom[row];
    // Don't change it now. We'll need it further.
    denom[row] *= pivot;

    // Eliminate pivot column in all other rows.
    for (int i = 0; i < tableau.size(); ++i) {
      if (i == row)
        continue;

      // We would write:
      //   auto factor = tableau[i][col];
      //   for (auto j = 0; j < width; ++j)
      //     tableau[i][j] -= factor * tableau[row][j];
      //
      // Then we're actually having
      //
      // tableau[i][j] = (tableau[i][j] / denom[i] - tableau[i][col] / denom[i] * tableau[row][j] / denom[row])
      //
      // To clear it up:
      //
      // tableau[i][j] = (tableau[i][j] * denom[row] - tableau[i][col] * tableau[row][j]) / (denom[i] * denom[row])
      //
      // So we just update according to that.

      denom[i] *= denom[row];
      for (auto j = 0; j < tableau[0].size(); ++j)
        tableau[i][j] = tableau[i][j] * denom[row] - tableau[i][col] * tableau[row][j];
    }

    // Calculate GCD to make numbers smaller.
    for (int i = 0; i < tableau.size(); i++) {
      auto &row = tableau[i];
      auto &de = denom[i];

      auto gcd = abs(de);
      for (auto x : row) {
        if ((gcd = std::gcd(gcd, abs(x))) == 1)
          break;
      }

      if (gcd != 1) {
        de /= gcd;
        for (auto &x : row)
          x /= gcd;
      }
    }
  }
}

void BasicSet::dump(std::ostream &os) {
  if (tableau.empty()) {
    os << "<no tableau>\n";
    return;
  }

  const int rows = tableau.size();
  const int cols = tableau[0].size();

  // Compute max width per column for alignment.
  std::vector<size_t> colWidths(cols, 0);
  for (int j = 0; j < cols; ++j) {
    size_t maxWidth = 0;
    for (int i = 0; i < rows; ++i) {
      auto num = tableau[i][j];
      auto den = denom[i];
      auto g = std::gcd(num, den);
      num /= g;
      den /= g;
      std::ostringstream ss;
      if (den == 1)
        ss << num;
      else
        ss << num << "/" << den;
      maxWidth = std::max(maxWidth, ss.str().size());
    }
    colWidths[j] = maxWidth;
  }

  // Print.
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      auto num = tableau[i][j];
      auto den = denom[i];
      auto g = std::gcd(num, den);
      num /= g;
      den /= g;

      std::ostringstream ss;
      if (den == 1)
        ss << num;
      else
        ss << num << "/" << den;

      os << std::setw(colWidths[j]) << ss.str() << (j + 1 < cols ? "  " : "");
    }
    os << "\n";
  }
}

BasicSet BasicSet::intersect(const BasicSet &other) const {
  std::vector<AffineExpr> combined;
  combined.reserve(tableau.size() + other.tableau.size());
  for (const auto &row : tableau)
    combined.push_back(row);
  for (const auto &row : other.tableau)
    combined.push_back(row);
  return BasicSet(combined);
}

bool BasicSet::isSubsetOf(const BasicSet &other) const {
  if (other.tableau.empty())
    return true;
  if (other.tableau.size() != 1)
    return false;
  if (!tableau.empty() && tableau[0].size() != other.tableau[0].size())
    return false;

  // Rows are interpreted as affine integer inequalities row(x) >= 0.
  // this ⊆ other iff this ∩ ¬other is empty. For integer domains,
  // ¬(row(x) >= 0) is row(x) <= -1, i.e. -row(x) - 1 >= 0.
  AffineExpr negated = other.tableau[0];
  for (auto &coeff : negated)
    coeff = -coeff;
  if (!negated.empty())
    negated.back() -= 1;

  BasicSet witness = intersect(BasicSet({negated}));
  return witness.empty();
}

int BasicSet::numVars() const {
  if (tableau.empty())
    return 0;
  return (int) tableau[0].size() - 1;
}

void BasicSet::addConstraint(const AffineExpr &row) {
  tableau.push_back(row);
}
