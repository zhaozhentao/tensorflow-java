package org.tensorflow.nio.buffer.impl.logical;

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer.LongMapper;
import org.tensorflow.nio.buffer.LongDataBufferTestBase;
import org.tensorflow.nio.buffer.impl.large.ByteLargeDataBuffer;

public class LongLogicalDataBufferTest extends LongDataBufferTestBase {

  @Override
  protected LongDataBuffer allocate(long capacity) {
    return DataBuffers.ofLongs(capacity, new TestLongMapper());
  }

  @Override
  protected long maxCapacity() {
    return ByteLargeDataBuffer.MAX_CAPACITY / 3;
  }

  private static class TestLongMapper implements LongMapper {

    @Override
    public void writeLong(ByteDataBuffer physicalBuffer, long value) {
      physicalBuffer.put((byte)(((value >> 56) & 0x80) | ((value >> 16) & 0x7F)));
      physicalBuffer.put((byte)((value >> 8) & 0xFF));
      physicalBuffer.put((byte)(value & 0xFF));
    }

    @Override
    public long readLong(ByteDataBuffer physicalBuffer) {
      long msb = physicalBuffer.get();
      long midb = physicalBuffer.get();
      long lsb = physicalBuffer.get();
      return ((msb & 0x80) << 56) | ((msb & 0x7F) << 16) | ((midb & 0xFF) << 8) | (lsb & 0xFF);
    }

    @Override
    public int sizeInBytes() {
      return 3;
    }
  }
}
