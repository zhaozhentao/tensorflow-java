package org.tensorflow.nio.buffer.impl.logical;

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.DoubleDataBuffer.DoubleMapper;
import org.tensorflow.nio.buffer.DoubleDataBufferTestBase;
import org.tensorflow.nio.buffer.impl.large.ByteLargeDataBuffer;

public class DoubleLogicalDataBufferTest extends DoubleDataBufferTestBase {

  @Override
  protected DoubleDataBuffer allocate(long capacity) {
    return DataBuffers.ofDoubles(capacity, new TestDoubleMapper());
  }

  @Override
  protected long maxCapacity() {
    return ByteLargeDataBuffer.MAX_CAPACITY / 3;
  }

  private static class TestDoubleMapper implements DoubleMapper {

    @Override
    public void writeDouble(ByteDataBuffer physicalBuffer, double value) {
      long bits = Double.doubleToLongBits(value);
      physicalBuffer.put((byte)((bits >> 56) & 0xFF));
      physicalBuffer.put((byte)((bits >> 48) & 0xFF));
      physicalBuffer.put((byte)((bits >> 40) & 0xFF));
    }

    @Override
    public double readDouble(ByteDataBuffer physicalBuffer) {
      long byte7 = physicalBuffer.get();
      long byte6 = physicalBuffer.get();
      long byte5 = physicalBuffer.get();
      return Double.longBitsToDouble(((byte7 & 0xFF) << 56) | ((byte6 & 0xFF) << 48) | ((byte5 & 0xFF) << 40));
    }

    @Override
    public int sizeInBytes() {
      return 3;
    }
  }
}
