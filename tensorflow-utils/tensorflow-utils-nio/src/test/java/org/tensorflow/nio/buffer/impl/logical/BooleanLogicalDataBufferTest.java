package org.tensorflow.nio.buffer.impl.logical;

import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.BooleanDataBuffer.BooleanMapper;
import org.tensorflow.nio.buffer.BooleanDataBufferTestBase;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.DoubleDataBuffer.DoubleMapper;
import org.tensorflow.nio.buffer.DoubleDataBufferTestBase;
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

  private static class TestBooleanMapper implements BooleanMapper {

    @Override
    public void writeBoolean(ByteDataBuffer physicalBuffer, boolean value) {
      physicalBuffer.put((byte)(value ? 1 : 0));
    }

    @Override
    public boolean readBoolean(ByteDataBuffer physicalBuffer) {
      return physicalBuffer.get() > 0;
    }

    @Override
    public int sizeInBytes() {
      return 1;
    }
  }
}
