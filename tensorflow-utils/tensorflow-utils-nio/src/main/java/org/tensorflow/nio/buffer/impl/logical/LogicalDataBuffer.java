package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.Stream;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.converter.DataConverter;

public class LogicalDataBuffer<T> extends AbstractLogicalDataBuffer<T, DataBuffer<T>> {

  public static <T> LogicalDataBuffer<T> map(ByteDataBuffer delegate, DataConverter<T> valueMapper) {
    return new LogicalDataBuffer<>(delegate, valueMapper);
  }

  @Override
  public Stream<T> stream() {
    throw new UnsupportedOperationException(); // FIXME TODO!
  }

  @Override
  public DataBuffer<T> duplicate() {
    return new LogicalDataBuffer<>(physicalBuffer().duplicate(), valueMapper());
  }

  private LogicalDataBuffer(ByteDataBuffer physicalBuffer, DataConverter<T> mapper) {
    super(physicalBuffer, mapper);
  }
}
