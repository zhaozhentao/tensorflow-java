package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

public interface BooleanDataConverter extends DataConverter<Boolean> {

  void writeBoolean(ByteDataBuffer physicalBuffer, boolean value);

  boolean readBoolean(ByteDataBuffer physicalBuffer);

  @Override
  default void writeValue(ByteDataBuffer physicalBuffer, Boolean value) {
    writeBoolean(physicalBuffer, value);
  }

  @Override
  default Boolean readValue(ByteDataBuffer physicalBuffer) {
    return readBoolean(physicalBuffer);
  }
}
