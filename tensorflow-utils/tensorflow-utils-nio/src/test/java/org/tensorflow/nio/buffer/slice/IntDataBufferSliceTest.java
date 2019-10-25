package org.tensorflow.nio.buffer.slice;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;

public class IntDataBufferSliceTest extends DataBufferSliceTestBase<Integer> {

  @Override
  protected Integer valueOf(Long val) {
    return val.intValue();
  }

  @Override
  protected DataBuffer<Integer> allocate(long capacity) {
    return DataBuffers.ofInts(capacity);
  }
}
