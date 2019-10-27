package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

/**
 * Converts a value of a given type to/from bytes
 *
 * @param <T> value type
 */
public interface DataConverter<T> {

  /**
   * Writes a value as bytes to the given buffer at its current position.
   *
   * @param buffer buffer that receives the value as bytes
   * @param value value
   */
  void writeValue(ByteDataBuffer buffer, T value);

  /**
   * Reads a value as bytes from the given buffer at its current position.
   *
   * @param buffer buffer that supplies the value as bytes
   * @return value
   */
  T readValue(ByteDataBuffer buffer);

  /**
   * Returns the number of bytes required to represent a single value
   */
  int sizeInBytes();
}
