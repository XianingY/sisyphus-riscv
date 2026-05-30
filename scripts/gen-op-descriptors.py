#!/usr/bin/env python3
"""Generate restricted MLIR-like op descriptors from the project YAML subset."""

import argparse
import pathlib
import re
import sys


REQUIRED_KEYS = ["name", "dialect", "operands", "results", "attrs", "traits", "verify", "fold"]
OPTIONAL_KEYS = ["cppClass", "typeFormat", "assemblyFormat", "interfaces"]
ALL_KEYS = set(REQUIRED_KEYS + OPTIONAL_KEYS)
ALLOWED_TRAITS = {
    "Pure",
    "MemoryEffect",
    "BranchLike",
    "LoopLike",
    "SameOperandsAndResultType",
    "Terminator",
    "HasRegion",
    "NoSideEffect",
    "Commutative",
    "IsolatedFromAbove",
    "Symbol",
    "FunctionLike",
    "AffineLike",
    "VectorLike",
    "MachineOp",
    "RegisterOp",
}
ALLOWED_INTERFACES = {
    "PureOpInterface",
    "MemoryEffectOpInterface",
    "RegionBranchOpInterface",
    "TerminatorOpInterface",
}


class SchemaError(Exception):
    pass


def parse_list(value, lineno):
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        raise SchemaError(f"line {lineno}: expected inline list, got {value!r}")
    body = value[1:-1].strip()
    if not body:
        return []
    items = []
    for item in body.split(","):
        item = item.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", item):
            raise SchemaError(f"line {lineno}: invalid list item {item!r}")
        items.append(item)
    return items


def parse_scalar(value, lineno):
    value = value.strip()
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value):
        raise SchemaError(f"line {lineno}: invalid scalar {value!r}")
    return value


def parse_schema(path):
    records = []
    current = None
    saw_ops = False
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "ops:":
            saw_ops = True
            continue
        if not saw_ops:
            raise SchemaError(f"line {lineno}: expected top-level 'ops:'")
        if stripped.startswith("- "):
            if current is not None:
                records.append(current)
            current = {}
            stripped = stripped[2:].strip()
            if not stripped:
                continue
        if ":" not in stripped:
            raise SchemaError(f"line {lineno}: expected key: value")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key not in ALL_KEYS:
            raise SchemaError(f"line {lineno}: unknown key {key!r}")
        if current is None:
            raise SchemaError(f"line {lineno}: field before first record")
        if key in current:
            raise SchemaError(f"line {lineno}: duplicate key {key!r}")
        if key in ("operands", "results", "attrs", "traits", "interfaces"):
            current[key] = parse_list(value, lineno)
        elif key in ("typeFormat", "assemblyFormat"):
            current[key] = value.strip()
        else:
            current[key] = parse_scalar(value, lineno)
    if current is not None:
        records.append(current)
    if not records:
        raise SchemaError("schema contains no ops")

    for rec in records:
        missing = [k for k in REQUIRED_KEYS if k not in rec]
        if missing:
            raise SchemaError(f"{rec.get('dialect', '?')}.{rec.get('name', '?')}: missing {missing[0]}")
        rec.setdefault("cppClass", class_name_for(rec))
        rec.setdefault("typeFormat", "legacy")
        rec.setdefault("assemblyFormat", "generic")
        rec.setdefault("interfaces", infer_interfaces(rec))
        for trait in rec["traits"]:
            if trait not in ALLOWED_TRAITS:
                raise SchemaError(f"{rec['dialect']}.{rec['name']}: unknown trait {trait}")
        for interface in rec["interfaces"]:
            if interface not in ALLOWED_INTERFACES:
                raise SchemaError(f"{rec['dialect']}.{rec['name']}: unknown interface {interface}")
    records.sort(key=lambda rec: (rec["dialect"], rec["name"]))
    for i in range(1, len(records)):
        if (records[i]["dialect"], records[i]["name"]) == (records[i - 1]["dialect"], records[i - 1]["name"]):
            raise SchemaError(f"duplicate op {records[i]['dialect']}.{records[i]['name']}")
    return records


def symbol_for(rec):
    return f"kTraits_{rec['dialect']}_{rec['name']}".replace("-", "_")


def names_symbol_for(rec, field):
    return f"k{field.title()}_{rec['dialect']}_{rec['name']}".replace("-", "_")


def class_name_for(rec):
    def cap(piece):
        return "".join(part[:1].upper() + part[1:] for part in piece.split("_"))
    return f"{cap(rec['dialect'])}{cap(rec['name'])}ODSOp"


def infer_interfaces(rec):
    interfaces = []
    traits = set(rec.get("traits", []))
    if "Pure" in traits or "NoSideEffect" in traits:
        interfaces.append("PureOpInterface")
    if "MemoryEffect" in traits:
        interfaces.append("MemoryEffectOpInterface")
    if "Terminator" in traits:
        interfaces.append("TerminatorOpInterface")
    if "BranchLike" in traits or "LoopLike" in traits or "HasRegion" in traits:
        interfaces.append("RegionBranchOpInterface")
    return interfaces


def method_suffix(name):
    return "".join(part[:1].upper() + part[1:] for part in name.split("_"))


def arity(values):
    if len(values) == 1 and values[0] == "variadic":
        return -1
    return len(values)


def render_descriptors(records):
    out = []
    out.append("// Generated by scripts/gen-op-descriptors.py. Do not edit by hand.")
    out.append("")
    out.append("namespace {")
    out.append("")
    for rec in records:
        for field in ("operands", "results", "attrs", "interfaces"):
            out.append(f"static const char *const {names_symbol_for(rec, field)}[] = {{")
            items = rec[field] if rec[field] else ["nullptr"]
            for item in items:
                if item == "nullptr":
                    out.append("  nullptr,")
                    continue
                out.append(f'  "{item}",')
            out.append("};")
            out.append("")
        out.append(f"static const char *const {symbol_for(rec)}[] = {{")
        for trait in rec["traits"]:
            out.append(f'  "{trait}",')
        out.append("};")
        out.append("")
    out.append("static const sys::ir::OpDescriptor kGeneratedOpDescriptors[] = {")
    for rec in records:
        out.append(
            '  {"%s", "%s", %d, %d, %d, "%s", "%s", %s, %d, %s, %d, %s, %d, %s, %d, "%s", "%s", "%s", %s, %d},'
            % (
                rec["dialect"],
                rec["name"],
                arity(rec["operands"]),
                arity(rec["results"]),
                arity(rec["attrs"]),
                rec["verify"],
                rec["fold"],
                symbol_for(rec),
                len(rec["traits"]),
                names_symbol_for(rec, "operands"),
                len(rec["operands"]),
                names_symbol_for(rec, "results"),
                len(rec["results"]),
                names_symbol_for(rec, "attrs"),
                len(rec["attrs"]),
                rec["cppClass"],
                rec["typeFormat"],
                rec["assemblyFormat"],
                names_symbol_for(rec, "interfaces"),
                len(rec["interfaces"]),
            )
        )
    out.append("};")
    out.append("")
    out.append("} // namespace")
    return "\n".join(out) + "\n"


def render_classes(records):
    out = []
    out.append("// Generated by scripts/gen-op-descriptors.py. Do not edit by hand.")
    out.append("#ifndef SISY_GENERATED_OP_CLASSES_INC")
    out.append("#define SISY_GENERATED_OP_CLASSES_INC")
    out.append("")
    out.append("namespace sys::ir::ods {")
    out.append("")
    for rec in records:
        cls = class_name_for(rec)
        out.append(f"class {cls} {{")
        out.append("  sys::ir::Operation *op = nullptr;")
        out.append("public:")
        out.append(f"  static constexpr const char *kDialect = \"{rec['dialect']}\";")
        out.append(f"  static constexpr const char *kName = \"{rec['name']}\";")
        out.append(f"  explicit {cls}(sys::ir::Operation *op): op(op) {{}}")
        out.append("  sys::ir::Operation *getOperation() const { return op; }")
        for idx, operand in enumerate(rec["operands"]):
            if operand == "variadic":
                continue
            out.append(f"  sys::Value get{method_suffix(operand)}() const {{ return op->getOperand({idx}); }}")
        out.append("  static sys::ir::Operation *build(sys::ir::Operation *op) { return op; }")
        out.append("  static bool verify(sys::ir::Operation *op, std::string *error = nullptr) {")
        out.append("    return op ? op->verifyBridge(error) : false;")
        out.append("  }")
        out.append("};")
        out.append("")
    out.append("} // namespace sys::ir::ods")
    out.append("")
    out.append("#endif")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("schema")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--check", action="store_true", help="compare generated output with an existing file")
    parser.add_argument("--emit", choices=("descriptors", "classes"), default="descriptors")
    args = parser.parse_args()

    try:
        records = parse_schema(pathlib.Path(args.schema))
        generated = render_descriptors(records) if args.emit == "descriptors" else render_classes(records)
        if args.check:
            if not args.output:
                raise SchemaError("--check requires an output file")
            existing = pathlib.Path(args.output).read_text()
            if existing != generated:
                sys.stderr.write(f"{args.output} is out of date; regenerate op descriptors\n")
                return 1
            return 0
        if args.output:
            pathlib.Path(args.output).write_text(generated)
        else:
            sys.stdout.write(generated)
        return 0
    except SchemaError as exc:
        sys.stderr.write(f"opgen error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
