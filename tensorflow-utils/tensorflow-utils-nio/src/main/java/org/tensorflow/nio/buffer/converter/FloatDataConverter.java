package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

/**
 * Converts a boolean to/from bytes
 */
public interface FloatDataConverter extends DataConverter<Float> {

  /**
   * Writes a float as bytes to the given buffer at its current position.
   *
   * @param buffer buffer that receives the value as bytes
   * @param value value
   */
  void writeFloat(ByteDataBuffer buffer, float value);

  /**
   * Reads a float as bytes from the given buffer at its current position.
   *
   * @param buffer buffer that supplies the value as bytes
   * @return value
   */
  float readFloat(ByteDataBuffer buffer);

  @Override
  default void writeValue(ByteDataBuffer buffer, Float value) {
    writeFloat(buffer, value);
  }

  @Override
  default Float readValue(ByteDataBuffer buffer) {
    return readFloat(buffer);
  }
}
