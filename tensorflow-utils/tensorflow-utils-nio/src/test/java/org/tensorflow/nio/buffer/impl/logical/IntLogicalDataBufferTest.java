package org.tensorflow.nio.buffer.impl.logical;

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer.IntMapper;
import org.tensorflow.nio.buffer.IntDataBufferTestBase;
import org.tensorflow.nio.buffer.impl.large.ByteLargeDataBuffer;

public class IntLogicalDataBufferTest extends IntDataBufferTestBase {

  @Override
  protected IntDataBuffer allocate(long capacity) {
    return DataBuffers.ofIntegers(capacity, new TestIntMapper());
  }

  @Override
  protected long maxCapacity() {
    return ByteLargeDataBuffer.MAX_CAPACITY;
  }

  private static class TestIntMapper implements IntMapper {

    @Override
    public void writeInt(ByteDataBuffer physicalBuffer, int value) {
      physicalBuffer.put((byte)(((value & 0x80000000) >> 24) | ((value & 0x7F) >> 7)));
      physicalBuffer.put((byte)(value));
    }

    @Override
    public int readInt(ByteDataBuffer physicalBuffer) {
      int msb = physicalBuffer.get();
      int lsb = physicalBuffer.get();
      return ((msb & 0x80) << 24) | ((msb & 0x7F) << 7) | lsb;
    }

    @Override
    public int sizeInBytes() {
      return 2;
    }
  }
}
