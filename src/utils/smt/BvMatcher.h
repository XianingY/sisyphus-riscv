#ifndef BV_MATCHER_H
#define BV_MATCHER_H

#include "BvExpr.h"
#include <vector>
#include <map>
#include <string>

namespace smt {

struct Expr {
  int id;
  Expr(int id): id(id) {}
  virtual ~Expr() {}
};

struct Atom : Expr {
  template<class T>
  static bool classof(T *t) { return t->id == 1; }

  std::string_view value;
  Atom(std::string_view value): Expr(1), value(value) {}
};

struct List : Expr {
  template<class T>
  static bool classof(T *t) { return t->id == 2; }

  std::vector<Expr*> elements;
  List(): Expr(2) {}
};

class BvRule {
  std::map<std::string_view, BvExpr*> binding;
  std::string_view text;
  std::vector<std::string> externalStrs;
  Expr *pattern;
  int loc = 0;
  bool failed = false;

  std::string_view nextToken();
  Expr *parse();

  bool matchExpr(Expr *expr, BvExpr *bvexpr);
  int evalExpr(Expr *expr);
  float evalFExpr(Expr *expr);
  BvExpr *buildExpr(Expr *expr);

  void dump(Expr *expr, std::ostream &os);
  void release(Expr *expr);
  BvExpr *rewriteRoot(BvExpr *expr);
public:
  using Binding = std::map<std::string, BvExpr*>;
  BvExprContext *ctx = nullptr;

  BvRule(const BvRule &other) = delete;

  BvRule(const char *text);
  ~BvRule();
  BvExpr *rewrite(BvExpr *expr);
  BvExpr *extract(const std::string &name);

  void dump(std::ostream &os = std::cerr);
};

}

#endif
