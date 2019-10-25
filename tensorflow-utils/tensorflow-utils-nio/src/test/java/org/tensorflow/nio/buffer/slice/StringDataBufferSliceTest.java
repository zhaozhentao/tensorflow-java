package org.tensorflow.nio.buffer.slice;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;

public class StringDataBufferSliceTest extends DataBufferSliceTestBase<String> {

  @Override
  protected String valueOf(Long val) {
    return val.toString();
  }

  @Override
  protected DataBuffer<String> allocate(long capacity) {
    return DataBuffers.of(String.class, capacity);
  }
}
