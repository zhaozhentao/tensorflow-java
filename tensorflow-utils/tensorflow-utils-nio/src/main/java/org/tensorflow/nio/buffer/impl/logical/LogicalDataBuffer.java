package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.Stream;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;

public class LogicalDataBuffer<T> extends AbstractLogicalDataBuffer<T, DataBuffer<T>> {

  public static <T> LogicalDataBuffer<T> map(ByteDataBuffer delegate, ValueMapper<T> valueMapper) {
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

  private LogicalDataBuffer(ByteDataBuffer physicalBuffer, ValueMapper<T> mapper) {
    super(physicalBuffer, mapper);
  }
}
