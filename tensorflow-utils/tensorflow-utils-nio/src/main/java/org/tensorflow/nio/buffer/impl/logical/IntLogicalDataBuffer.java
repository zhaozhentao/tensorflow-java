package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.IntStream;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.buffer.converter.IntDataConverter;
import org.tensorflow.nio.buffer.impl.Validator;

public class IntLogicalDataBuffer extends AbstractLogicalDataBuffer<Integer, IntDataBuffer>
    implements IntDataBuffer {

  public static IntLogicalDataBuffer map(ByteDataBuffer delegate, IntDataConverter intMapper) {
    return new IntLogicalDataBuffer(delegate, intMapper);
  }

  @Override
  public IntStream intStream() {
    return IntStream.iterate(0, this::get).limit(remaining());
  }

  @Override
  public int getInt() {
    return converter.readInt(physicalBuffer());
  }

  @Override
  public int getInt(long index) {
    Validator.getArgs(this, index);
    return converter.readInt(physicalBuffer().withPosition(index * converter.sizeInBytes()));
  }

  @Override
  public IntDataBuffer get(int[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = converter.readInt(physicalBuffer());
    }
    return this;
  }

  @Override
  public IntDataBuffer putInt(int value) {
    converter.writeInt(physicalBuffer(), value);
    return this;
  }

  @Override
  public IntDataBuffer putInt(long index, int value) {
    Validator.putArgs(this, index);
    converter.writeInt(physicalBuffer().withPosition(index * converter.sizeInBytes()), value);
    return this;
  }

  @Override
  public IntDataBuffer put(int[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      converter.writeInt(physicalBuffer(), src[i]);
    }
    return this;
  }

  @Override
  public IntDataBuffer duplicate() {
    return new IntLogicalDataBuffer(physicalBuffer().duplicate(), converter);
  }

  private IntLogicalDataBuffer(ByteDataBuffer physicalBuffer, IntDataConverter converter) {
    super(physicalBuffer, converter);
    this.converter = converter;
  }

  private IntDataConverter converter;
}
