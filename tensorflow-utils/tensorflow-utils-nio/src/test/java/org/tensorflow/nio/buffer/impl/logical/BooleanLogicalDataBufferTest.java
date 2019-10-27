package org.tensorflow.nio.buffer.impl.logical;

import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.BooleanDataBufferTestBase;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.converter.BooleanDataConverter;
import org.tensorflow.nio.buffer.impl.large.ByteLargeDataBuffer;

public class BooleanLogicalDataBufferTest extends BooleanDataBufferTestBase {

  @Override
  protected BooleanDataBuffer allocate(long capacity) {
    return DataBuffers.ofBooleans(capacity, new TestBooleanMapper());
  }

  @Override
  protected long maxCapacity() {
    return ByteLargeDataBuffer.MAX_CAPACITY;
  }

  private static class TestBooleanMapper implements BooleanDataConverter {

    @Override
    public void writeBoolean(ByteDataBuffer buffer, boolean value) {
      buffer.put((byte)(value ? 1 : 0));
    }

    @Override
    public boolean readBoolean(ByteDataBuffer buffer) {
      return buffer.get() > 0;
    }

    @Override
    public int sizeInBytes() {
      return 1;
    }
  }
}
