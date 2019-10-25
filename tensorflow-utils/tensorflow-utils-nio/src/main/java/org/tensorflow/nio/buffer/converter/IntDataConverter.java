package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

public interface IntDataConverter extends DataConverter<Integer> {

  void writeInt(ByteDataBuffer physicalBuffer, int value);

  int readInt(ByteDataBuffer physicalBuffer);

  @Override
  default void writeValue(ByteDataBuffer physicalBuffer, Integer value) {
    writeInt(physicalBuffer, value);
  }

  @Override
  default Integer readValue(ByteDataBuffer physicalBuffer) {
    return readInt(physicalBuffer);
  }
}
