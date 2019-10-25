package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

public interface DoubleDataConverter extends DataConverter<Double> {

  void writeDouble(ByteDataBuffer physicalBuffer, double value);

  double readDouble(ByteDataBuffer physicalBuffer);

  @Override
  default void writeValue(ByteDataBuffer physicalBuffer, Double value) {
    writeDouble(physicalBuffer, value);
  }

  @Override
  default Double readValue(ByteDataBuffer physicalBuffer) {
    return readDouble(physicalBuffer);
  }
}
