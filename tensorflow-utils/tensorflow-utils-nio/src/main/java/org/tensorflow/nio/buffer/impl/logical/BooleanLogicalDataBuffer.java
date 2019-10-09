package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.Stream;
import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

public class BooleanLogicalDataBuffer extends AbstractLogicalDataBuffer<Boolean, BooleanDataBuffer>
    implements BooleanDataBuffer {

  public static BooleanLogicalDataBuffer map(ByteDataBuffer delegate, BooleanMapper booleanMapper) {
    return new BooleanLogicalDataBuffer(delegate, booleanMapper);
  }

  @Override
  public BooleanDataBuffer get(boolean[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = booleanMapper.readBoolean(physicalBuffer());
    }
    return this;
  }

  @Override
  public BooleanDataBuffer put(boolean[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      booleanMapper.writeBoolean(physicalBuffer(), src[i]);
    }
    return this;
  }

  @Override
  public Stream<Boolean> stream() {
    throw new UnsupportedOperationException("BooleanDataBuffer does not support value streaming at the moment");
  }

  @Override
  public BooleanDataBuffer duplicate() {
    return new BooleanLogicalDataBuffer(physicalBuffer().duplicate(), booleanMapper);
  }

  private BooleanLogicalDataBuffer(ByteDataBuffer physicalBuffer, BooleanMapper booleanMapper) {
    super(physicalBuffer, booleanMapper);
    this.booleanMapper = booleanMapper;
  }

  private BooleanMapper booleanMapper;
}
