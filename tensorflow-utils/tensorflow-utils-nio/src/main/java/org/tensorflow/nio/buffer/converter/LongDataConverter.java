package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

public interface LongDataConverter extends DataConverter<Long> {

  void writeLong(ByteDataBuffer physicalBuffer, long value);

  long readLong(ByteDataBuffer physicalBuffer);

  @Override
  default void writeValue(ByteDataBuffer physicalBuffer, Long value) {
    writeLong(physicalBuffer, value);
  }

  @Override
  default Long readValue(ByteDataBuffer physicalBuffer) {
    return readLong(physicalBuffer);
  }
}
