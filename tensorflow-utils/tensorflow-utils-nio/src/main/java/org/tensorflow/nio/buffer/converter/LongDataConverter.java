package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

/**
 * Converts a boolean to/from bytes
 */
public interface LongDataConverter extends DataConverter<Long> {

  /**
   * Writes a long as bytes to the given buffer at its current position.
   *
   * @param buffer buffer that receives the value as bytes
   * @param value value
   */
  void writeLong(ByteDataBuffer buffer, long value);

  /**
   * Reads a long as bytes from the given buffer at its current position.
   *
   * @param buffer buffer that supplies the value as bytes
   * @return value
   */
  long readLong(ByteDataBuffer buffer);

  @Override
  default void writeValue(ByteDataBuffer buffer, Long value) {
    writeLong(buffer, value);
  }

  @Override
  default Long readValue(ByteDataBuffer buffer) {
    return readLong(buffer);
  }
}
