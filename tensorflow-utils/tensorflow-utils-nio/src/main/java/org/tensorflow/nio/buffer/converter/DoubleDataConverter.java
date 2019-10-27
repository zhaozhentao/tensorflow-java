package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

/**
 * Converts a boolean to/from bytes
 */
public interface DoubleDataConverter extends DataConverter<Double> {

  /**
   * Writes a double as bytes to the given buffer at its current position.
   *
   * @param buffer buffer that receives the value as bytes
   * @param value value
   */
  void writeDouble(ByteDataBuffer buffer, double value);

  /**
   * Reads a double as bytes from the given buffer at its current position.
   *
   * @param buffer buffer that supplies the value as bytes
   * @return value
   */
  double readDouble(ByteDataBuffer buffer);

  @Override
  default void writeValue(ByteDataBuffer buffer, Double value) {
    writeDouble(buffer, value);
  }

  @Override
  default Double readValue(ByteDataBuffer buffer) {
    return readDouble(buffer);
  }
}
