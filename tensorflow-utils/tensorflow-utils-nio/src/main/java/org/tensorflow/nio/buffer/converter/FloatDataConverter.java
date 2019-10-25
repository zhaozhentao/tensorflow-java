package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

public interface FloatDataConverter extends DataConverter<Float> {

  void writeFloat(ByteDataBuffer physicalBuffer, float value);

  float readFloat(ByteDataBuffer physicalBuffer);

  @Override
  default void writeValue(ByteDataBuffer physicalBuffer, Float value) {
    writeFloat(physicalBuffer, value);
  }

  @Override
  default Float readValue(ByteDataBuffer physicalBuffer) {
    return readFloat(physicalBuffer);
  }
}
