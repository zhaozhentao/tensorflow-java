package org.tensorflow.nio.buffer.impl.logical;

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.IntDataBufferTestBase;
import org.tensorflow.nio.buffer.converter.IntDataConverter;
import org.tensorflow.nio.buffer.impl.large.ByteLargeDataBuffer;

public class IntLogicalDataBufferTest extends IntDataBufferTestBase {

  @Override
  protected IntDataBuffer allocate(long capacity) {
    return DataBuffers.ofInts(capacity, new TestIntMapper());
  }

  @Override
  protected long maxCapacity() {
    return ByteLargeDataBuffer.MAX_CAPACITY;
  }

  private static class TestIntMapper implements IntDataConverter {

    @Override
    public void writeInt(ByteDataBuffer buffer, int value) {
      buffer.put((byte)(((value & 0x80000000) >> 24) | ((value & 0x7F) >> 7)));
      buffer.put((byte)(value));
    }

    @Override
    public int readInt(ByteDataBuffer buffer) {
      int msb = buffer.get();
      int lsb = buffer.get();
      return ((msb & 0x80) << 24) | ((msb & 0x7F) << 7) | lsb;
    }

    @Override
    public int sizeInBytes() {
      return 2;
    }
  }
}
