#include "DataLayout.h"

using namespace sys;

std::size_t DataLayout::sizeOf(Value::Type type) const {
  switch (type) {
  case Value::unit:
    return 0;
  case Value::i32:
  case Value::f32:
    return 4;
  case Value::i64:
    return pointerBytes;
  case Value::i128:
  case Value::f128:
    return 16;
  case Value::vscale_i32:
  case Value::vscale_f32:
    return 0;
  }
  return 0;
}

std::size_t DataLayout::alignmentOf(Value::Type type) const {
  auto size = sizeOf(type);
  if (size == 0)
    return 1;
  return size > pointerBytes ? pointerBytes : size;
}
