package org.tensorflow.nio.buffer.slice;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;

public class BooleanDataBufferSliceTest extends DataBufferSliceTestBase<Boolean> {

  @Override
  protected Boolean valueOf(Long val) {
    return val > 0;
  }

  @Override
  protected DataBuffer<Boolean> allocate(long capacity) {
    return DataBuffers.ofBooleans(capacity);
  }
}
