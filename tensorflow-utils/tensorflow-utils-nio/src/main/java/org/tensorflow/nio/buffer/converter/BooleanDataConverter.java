package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

/**
 * Converts a boolean to/from bytes
 */
public interface BooleanDataConverter extends DataConverter<Boolean> {

  /**
   * Writes a boolean as bytes to the given buffer at its current position.
   *
   * @param buffer buffer that receives the value as bytes
   * @param value value
   */
  void writeBoolean(ByteDataBuffer buffer, boolean value);

  /**
   * Reads a boolean as bytes from the given buffer at its current position.
   *
   * @param buffer buffer that supplies the value as bytes
   * @return value
   */
  boolean readBoolean(ByteDataBuffer buffer);

  @Override
  default void writeValue(ByteDataBuffer buffer, Boolean value) {
    writeBoolean(buffer, value);
  }

  @Override
  default Boolean readValue(ByteDataBuffer buffer) {
    return readBoolean(buffer);
  }
}
