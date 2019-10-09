package org.tensorflow.nio.buffer.impl.logical;

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer.FloatMapper;
import org.tensorflow.nio.buffer.FloatDataBufferTestBase;
import org.tensorflow.nio.buffer.impl.large.ByteLargeDataBuffer;

public class FloatLogicalDataBufferTest extends FloatDataBufferTestBase {

  @Override
  protected FloatDataBuffer allocate(long capacity) {
    return DataBuffers.ofFloats(capacity, new TestFloatMapper());
  }

  @Override
  protected long maxCapacity() {
    return ByteLargeDataBuffer.MAX_CAPACITY / 2;
  }

  private static class TestFloatMapper implements FloatMapper {

    @Override
    public void writeFloat(ByteDataBuffer physicalBuffer, float value) {
      int bits = Float.floatToIntBits(value);
      physicalBuffer.put((byte)((bits >> 24) & 0xFF));
      physicalBuffer.put((byte)((bits >> 16) & 0xFF));
    }

    @Override
    public float readFloat(ByteDataBuffer physicalBuffer) {
      int byte3 = physicalBuffer.get();
      int byte2 = physicalBuffer.get();
      return Float.intBitsToFloat(((byte3 & 0xFF) << 24) | ((byte2 & 0xFF) << 16));
    }

    @Override
    public int sizeInBytes() {
      return 2;
    }
  }
}
