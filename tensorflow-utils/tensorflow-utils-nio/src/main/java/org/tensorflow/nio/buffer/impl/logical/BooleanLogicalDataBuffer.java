package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.Stream;
import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.converter.BooleanDataConverter;
import org.tensorflow.nio.buffer.impl.Validator;

public class BooleanLogicalDataBuffer extends AbstractLogicalDataBuffer<Boolean, BooleanDataBuffer>
    implements BooleanDataBuffer {

  public static BooleanLogicalDataBuffer map(ByteDataBuffer delegate, BooleanDataConverter booleanMapper) {
    return new BooleanLogicalDataBuffer(delegate, booleanMapper);
  }

  @Override
  public boolean getBoolean() {
    return converter.readBoolean(physicalBuffer());
  }

  @Override
  public boolean getBoolean(long index) {
    Validator.getArgs(this, index);
    return converter.readBoolean(physicalBuffer().withPosition(index * converter.sizeInBytes()));
  }

  @Override
  public BooleanDataBuffer get(boolean[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = converter.readBoolean(physicalBuffer());
    }
    return this;
  }

  @Override
  public BooleanDataBuffer putBoolean(boolean value) {
    converter.writeBoolean(physicalBuffer(), value);
    return this;
  }

  @Override
  public BooleanDataBuffer putBoolean(long index, boolean value) {
    Validator.putArgs(this, index);
    converter.writeBoolean(physicalBuffer().withPosition(index * converter.sizeInBytes()), value);
    return this;
  }

  @Override
  public BooleanDataBuffer put(boolean[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      converter.writeBoolean(physicalBuffer(), src[i]);
    }
    return this;
  }

  @Override
  public Stream<Boolean> stream() {
    throw new UnsupportedOperationException("BooleanDataBuffer does not support value streaming at the moment");
  }

  @Override
  public BooleanDataBuffer duplicate() {
    return new BooleanLogicalDataBuffer(physicalBuffer().duplicate(), converter);
  }

  private BooleanLogicalDataBuffer(ByteDataBuffer physicalBuffer, BooleanDataConverter converter) {
    super(physicalBuffer, converter);
    this.converter = converter;
  }

  private BooleanDataConverter converter;
}
