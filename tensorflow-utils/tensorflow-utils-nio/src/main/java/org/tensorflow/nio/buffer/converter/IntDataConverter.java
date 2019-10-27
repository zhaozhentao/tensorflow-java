package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

/**
 * Converts a boolean to/from bytes
 */
public interface IntDataConverter extends DataConverter<Integer> {

  /**
   * Writes an integer as bytes to the given buffer at its current position.
   *
   * @param buffer buffer that receives the value as bytes
   * @param value value
   */
  void writeInt(ByteDataBuffer buffer, int value);

  /**
   * Reads an integer as bytes from the given buffer at its current position.
   *
   * @param buffer buffer that supplies the value as bytes
   * @return value
   */
  int readInt(ByteDataBuffer buffer);

  @Override
  default void writeValue(ByteDataBuffer buffer, Integer value) {
    writeInt(buffer, value);
  }

  @Override
  default Integer readValue(ByteDataBuffer buffer) {
    return readInt(buffer);
  }
}
