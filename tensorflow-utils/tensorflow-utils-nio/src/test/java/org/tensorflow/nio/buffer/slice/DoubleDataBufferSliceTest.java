package org.tensorflow.nio.buffer.slice;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;

public class DoubleDataBufferSliceTest extends DataBufferSliceTestBase<Double> {

  @Override
  protected Double valueOf(Long val) {
    return val.doubleValue();
  }

  @Override
  protected DataBuffer<Double> allocate(long capacity) {
    return DataBuffers.ofDoubles(capacity);
  }
}
