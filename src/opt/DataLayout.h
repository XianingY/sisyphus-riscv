#ifndef SISY_DATA_LAYOUT_H
#define SISY_DATA_LAYOUT_H

#include "../codegen/OpBase.h"

#include <cstddef>

namespace sys {

class DataLayout {
  std::size_t pointerBytes = 8;

public:
  explicit DataLayout(std::size_t pointerBytes = 8): pointerBytes(pointerBytes) {}

  std::size_t pointerSize() const { return pointerBytes; }
  std::size_t sizeOf(Value::Type type) const;
  std::size_t alignmentOf(Value::Type type) const;
};

} // namespace sys

#endif
