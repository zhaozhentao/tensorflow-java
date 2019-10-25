package org.tensorflow.nio.buffer.converter;

import org.tensorflow.nio.buffer.ByteDataBuffer;

public interface DataConverter<T> {

  void writeValue(ByteDataBuffer physicalBuffer, T value);

  T readValue(ByteDataBuffer physicalBuffer);

  int sizeInBytes();
}
