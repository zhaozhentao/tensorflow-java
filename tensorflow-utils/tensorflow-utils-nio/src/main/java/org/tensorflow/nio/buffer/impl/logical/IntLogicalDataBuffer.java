package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.IntStream;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

public class IntLogicalDataBuffer extends AbstractLogicalDataBuffer<Integer, IntDataBuffer>
    implements IntDataBuffer {

  public static IntLogicalDataBuffer map(ByteDataBuffer delegate, IntMapper intMapper) {
    return new IntLogicalDataBuffer(delegate, intMapper);
  }

  @Override
  public IntStream intStream() {
    return IntStream.iterate(0, this::get).limit(remaining());
  }

  @Override
  public IntDataBuffer get(int[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = intMapper.readInt(physicalBuffer());
    }
    return this;
  }

  @Override
  public IntDataBuffer put(int[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      intMapper.writeInt(physicalBuffer(), src[i]);
    }
    return this;
  }

  @Override
  public IntDataBuffer duplicate() {
    return new IntLogicalDataBuffer(physicalBuffer().duplicate(), intMapper);
  }

  private IntLogicalDataBuffer(ByteDataBuffer physicalBuffer, IntMapper intMapper) {
    super(physicalBuffer, intMapper);
    this.intMapper = intMapper;
  }

  private IntMapper intMapper;
}
