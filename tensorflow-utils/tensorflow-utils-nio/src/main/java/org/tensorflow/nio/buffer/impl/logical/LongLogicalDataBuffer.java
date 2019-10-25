package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.LongStream;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.buffer.converter.LongDataConverter;
import org.tensorflow.nio.buffer.impl.Validator;

public class LongLogicalDataBuffer extends AbstractLogicalDataBuffer<Long, LongDataBuffer>
    implements LongDataBuffer {

  public static LongLogicalDataBuffer map(ByteDataBuffer delegate, LongDataConverter longMapper) {
    return new LongLogicalDataBuffer(delegate, longMapper);
  }

  @Override
  public LongStream longStream() {
    return LongStream.iterate(0, this::get).limit(remaining());
  }

  @Override
  public long getLong() {
    return converter.readLong(physicalBuffer());
  }

  @Override
  public long getLong(long index) {
    Validator.getArgs(this, index);
    return converter.readLong(physicalBuffer().withPosition(index * converter.sizeInBytes()));
  }

  @Override
  public LongDataBuffer get(long[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = converter.readLong(physicalBuffer());
    }
    return this;
  }

  @Override
  public LongDataBuffer putLong(long value) {
    converter.writeLong(physicalBuffer(), value);
    return this;
  }

  @Override
  public LongDataBuffer putLong(long index, long value) {
    Validator.putArgs(this, index);
    converter.writeLong(physicalBuffer().withPosition(index * converter.sizeInBytes()), value);
    return this;
  }

  @Override
  public LongDataBuffer put(long[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      converter.writeLong(physicalBuffer(), src[i]);
    }
    return this;
  }

  @Override
  public LongDataBuffer duplicate() {
    return new LongLogicalDataBuffer(physicalBuffer().duplicate(), converter);
  }

  private LongLogicalDataBuffer(ByteDataBuffer physicalBuffer, LongDataConverter converter) {
    super(physicalBuffer, converter);
    this.converter = converter;
  }

  private LongDataConverter converter;
}
