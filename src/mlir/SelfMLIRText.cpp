#include "SelfMLIR.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace sys::mlir {

static std::string stripQuotes(const std::string &text) {
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
    return text.substr(1, text.size() - 2);
  return text;
}

static void printValueList(const std::vector<Value> &values, std::ostream &os) {
  for (size_t i = 0; i < values.size(); i++) {
    if (i)
      os << ", ";
    os << values[i].printName() << " : " << values[i].type().str();
  }
}

static void printOp(Operation &op, std::ostream &os, int indent) {
  std::string pad(indent, ' ');
  if (op.isErased())
    return;
  os << pad;
  if (op.resultCount() == 1)
    os << op.result().printName() << " = ";
  os << '"' << op.name() << '"';
  os << "(";
  printValueList(op.getOperands(), os);
  os << ")";
  if (!op.attrs().empty()) {
    os << " {";
    bool first = true;
    for (const auto &kv : op.attrs()) {
      if (!first)
        os << ", ";
      first = false;
      os << kv.first << " = " << kv.second.str();
    }
    os << "}";
  }
  if (op.resultCount() > 0) {
    os << " -> (";
    for (int i = 0; i < op.resultCount(); i++) {
      if (i)
        os << ", ";
      os << op.resultType(i).str();
    }
    os << ")";
  }
  os << " " << op.loc().str();
  if (op.getRegions().empty()) {
    os << "\n";
    return;
  }
  os << " {\n";
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      os << pad << "  ^bb";
      os << (reinterpret_cast<std::uintptr_t>(block.get()) & 0xffff) << "(";
      for (size_t i = 0; i < block->args().size(); i++) {
        if (i)
          os << ", ";
        auto &arg = *block->args()[i];
        os << "%" << arg.name() << " : " << arg.type().str();
      }
      os << "):\n";
      for (auto &child : block->ops())
        printOp(*child, os, indent + 4);
    }
  }
  os << pad << "}\n";
}

void print(Module &module, std::ostream &os) {
  printOp(module.op(), os, 0);
}

static std::string trim(const std::string &s) {
  size_t begin = 0;
  while (begin < s.size() && std::isspace((unsigned char) s[begin]))
    begin++;
  size_t end = s.size();
  while (end > begin && std::isspace((unsigned char) s[end - 1]))
    end--;
  return s.substr(begin, end - begin);
}

static std::vector<std::string> splitTopLevel(const std::string &text, char sep) {
  std::vector<std::string> out;
  std::string cur;
  bool inString = false;
  int parens = 0;
  int angles = 0;
  for (char c : text) {
    if (c == '"')
      inString = !inString;
    else if (!inString) {
      if (c == '(')
        parens++;
      else if (c == ')' && parens > 0)
        parens--;
      else if (c == '<')
        angles++;
      else if (c == '>' && angles > 0)
        angles--;
      else if (c == sep && parens == 0 && angles == 0) {
        out.push_back(trim(cur));
        cur.clear();
        continue;
      }
    }
    cur.push_back(c);
  }
  if (!trim(cur).empty())
    out.push_back(trim(cur));
  return out;
}

static Type parseType(Context &ctx, const std::string &text) {
  std::string ty = trim(text);
  if (ty == "none")
    return ctx.noneType();
  if (ty == "index")
    return ctx.index();
  if (ty.size() > 1 && ty[0] == 'i' &&
      std::all_of(ty.begin() + 1, ty.end(), [](char c) { return std::isdigit((unsigned char) c); }))
    return ctx.i((unsigned) std::stoul(ty.substr(1)));
  if (ty.size() > 1 && ty[0] == 'f' &&
      std::all_of(ty.begin() + 1, ty.end(), [](char c) { return std::isdigit((unsigned char) c); }))
    return ctx.f((unsigned) std::stoul(ty.substr(1)));
  if (ty.rfind("!riscv.reg<", 0) == 0)
    return ctx.reg("riscv", ty.substr(11, ty.size() > 12 ? ty.size() - 12 : 0));
  if (ty.rfind("!arm.reg<", 0) == 0)
    return ctx.reg("arm", ty.substr(9, ty.size() > 10 ? ty.size() - 10 : 0));
  return ctx.noneType();
}

static Attribute parseAttribute(Context &ctx, const std::string &text) {
  std::string value = trim(text);
  if (value == "true" || value == "false")
    return ctx.boolAttr(value == "true");
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
    return ctx.stringAttr(stripQuotes(value));
  size_t colon = value.find(':');
  std::string number = trim(colon == std::string::npos ? value : value.substr(0, colon));
  if (!number.empty()) {
    size_t pos = (number[0] == '-' || number[0] == '+') ? 1 : 0;
    bool allDigits = pos < number.size();
    for (; pos < number.size(); pos++)
      allDigits = allDigits && std::isdigit((unsigned char) number[pos]);
    if (allDigits) {
      Type type = colon == std::string::npos ? ctx.i(32) : parseType(ctx, value.substr(colon + 1));
      return ctx.integerAttr(std::stoll(number), type);
    }
  }
  return ctx.stringAttr(value);
}

static Location parseLocation(Context &ctx, const std::string &line) {
  size_t loc = line.find("loc(\"");
  if (loc == std::string::npos)
    return ctx.unknownLoc();
  size_t fileBegin = loc + 5;
  size_t fileEnd = line.find("\":", fileBegin);
  if (fileEnd == std::string::npos)
    return ctx.unknownLoc();
  size_t lineBegin = fileEnd + 2;
  size_t lineEnd = line.find(':', lineBegin);
  size_t colEnd = line.find(')', lineEnd == std::string::npos ? lineBegin : lineEnd + 1);
  if (lineEnd == std::string::npos || colEnd == std::string::npos)
    return ctx.unknownLoc();
  int parsedLine = std::atoi(line.substr(lineBegin, lineEnd - lineBegin).c_str());
  int parsedCol = std::atoi(line.substr(lineEnd + 1, colEnd - lineEnd - 1).c_str());
  return ctx.loc(line.substr(fileBegin, fileEnd - fileBegin), parsedLine, parsedCol);
}

static std::map<std::string, Attribute> parseAttrs(Context &ctx, const std::string &line) {
  std::map<std::string, Attribute> attrs;
  size_t open = line.find('{');
  size_t arrow = line.find("->");
  size_t loc = line.find(" loc(");
  if (open == std::string::npos || (arrow != std::string::npos && arrow < open) ||
      (loc != std::string::npos && loc < open))
    return attrs;
  size_t close = line.find('}', open);
  if (close == std::string::npos)
    return attrs;
  std::string body = line.substr(open + 1, close - open - 1);
  for (const auto &entry : splitTopLevel(body, ',')) {
    size_t eq = entry.find('=');
    if (eq == std::string::npos)
      continue;
    attrs[trim(entry.substr(0, eq))] = parseAttribute(ctx, entry.substr(eq + 1));
  }
  return attrs;
}

static std::vector<Type> parseResultTypes(Context &ctx, const std::string &line) {
  std::vector<Type> types;
  size_t arrow = line.find("->");
  if (arrow == std::string::npos)
    return types;
  size_t open = line.find('(', arrow);
  size_t close = line.find(')', open == std::string::npos ? arrow : open + 1);
  if (open == std::string::npos || close == std::string::npos)
    return types;
  for (const auto &part : splitTopLevel(line.substr(open + 1, close - open - 1), ','))
    types.push_back(parseType(ctx, part));
  return types;
}

std::unique_ptr<Module> parse(Context &ctx, const std::string &text,
                              std::vector<std::string> &errors) {
  auto module = std::make_unique<Module>(ctx);
  std::vector<Region*> regionStack{&module->body()};
  Block *currentBlock = module->body().getBlocks()[0].get();
  std::map<std::string, Value> values;

  std::istringstream input(text);
  std::string raw;
  int lineNo = 0;
  while (std::getline(input, raw)) {
    lineNo++;
    std::string line = trim(raw);
    if (line.empty())
      continue;
    if (line == "}") {
      if (regionStack.size() > 1) {
        regionStack.pop_back();
        auto &blocks = regionStack.back()->getBlocks();
        currentBlock = blocks.empty() ? nullptr : blocks.back().get();
      }
      continue;
    }
    if (line.rfind("\"builtin.module\"", 0) == 0)
      continue;
    if (line.rfind("^bb", 0) == 0) {
      Region *region = regionStack.back();
      bool useExisting = region == &module->body() && region->getBlocks().size() == 1 &&
                         region->getBlocks()[0]->ops().empty() &&
                         region->getBlocks()[0]->args().empty();
      currentBlock = useExisting ? region->getBlocks()[0].get() : &region->addBlock();
      size_t open = line.find('(');
      size_t close = line.find(')', open == std::string::npos ? 0 : open + 1);
      if (open != std::string::npos && close != std::string::npos) {
        for (const auto &argText : splitTopLevel(line.substr(open + 1, close - open - 1), ',')) {
          size_t colon = argText.rfind(':');
          if (colon == std::string::npos)
            continue;
          std::string name = trim(argText.substr(0, colon));
          if (!name.empty() && name.front() == '%')
            name.erase(name.begin());
          auto &arg = currentBlock->addArgument(parseType(ctx, argText.substr(colon + 1)),
                                                ctx.unknownLoc(), name);
          values["%" + arg.name()] = arg.value();
        }
      }
      continue;
    }

    if (!currentBlock) {
      errors.push_back("line " + std::to_string(lineNo) + ": operation outside block");
      continue;
    }
    std::string resultName;
    size_t quote = line.find('"');
    size_t equals = line.find('=');
    if (equals != std::string::npos && equals < quote) {
      resultName = trim(line.substr(0, equals));
      quote = line.find('"', equals);
    }
    size_t quoteEnd = line.find('"', quote + 1);
    if (quote == std::string::npos || quoteEnd == std::string::npos) {
      errors.push_back("line " + std::to_string(lineNo) + ": expected quoted operation name");
      continue;
    }
    std::string opName = line.substr(quote + 1, quoteEnd - quote - 1);
    size_t operandsOpen = line.find('(', quoteEnd);
    size_t operandsClose = line.find(')', operandsOpen == std::string::npos ? quoteEnd : operandsOpen + 1);
    std::vector<Value> operands;
    if (operandsOpen != std::string::npos && operandsClose != std::string::npos) {
      for (const auto &operandText : splitTopLevel(line.substr(operandsOpen + 1,
                                                               operandsClose - operandsOpen - 1), ',')) {
        size_t colon = operandText.rfind(':');
        std::string name = trim(colon == std::string::npos ? operandText
                                                           : operandText.substr(0, colon));
        auto it = values.find(name);
        if (it != values.end())
          operands.push_back(it->second);
        else if (!name.empty())
          errors.push_back("line " + std::to_string(lineNo) + ": unknown SSA value " + name);
      }
    }
    auto &op = Builder(ctx, currentBlock).create(opName, operands, parseResultTypes(ctx, line),
                                                 parseAttrs(ctx, line), parseLocation(ctx, line));
    if (!resultName.empty() && op.resultCount() == 1)
      values[resultName] = op.result();
    if (!line.empty() && line.back() == '{') {
      Region &region = op.addRegion();
      regionStack.push_back(&region);
      currentBlock = nullptr;
    }
  }
  return module;
}


} // namespace sys::mlir
