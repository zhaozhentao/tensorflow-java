package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.LongStream;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

public class LongLogicalDataBuffer extends AbstractLogicalDataBuffer<Long, LongDataBuffer>
    implements LongDataBuffer {

  public static LongLogicalDataBuffer map(ByteDataBuffer delegate, LongMapper longMapper) {
    return new LongLogicalDataBuffer(delegate, longMapper);
  }

  @Override
  public LongStream longStream() {
    return LongStream.iterate(0, this::get).limit(remaining());
  }

  @Override
  public LongDataBuffer get(long[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = longMapper.readLong(physicalBuffer());
    }
    return this;
  }

  @Override
  public LongDataBuffer put(long[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      longMapper.writeLong(physicalBuffer(), src[i]);
    }
    return this;
  }

  @Override
  public LongDataBuffer duplicate() {
    return new LongLogicalDataBuffer(physicalBuffer().duplicate(), longMapper);
  }

  private LongLogicalDataBuffer(ByteDataBuffer physicalBuffer, LongMapper longMapper) {
    super(physicalBuffer, longMapper);
    this.longMapper = longMapper;
  }

  private LongMapper longMapper;
}
