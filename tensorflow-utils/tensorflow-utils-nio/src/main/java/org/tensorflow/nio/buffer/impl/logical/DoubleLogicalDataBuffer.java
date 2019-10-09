package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.DoubleStream;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

public class DoubleLogicalDataBuffer extends AbstractLogicalDataBuffer<Double, DoubleDataBuffer>
    implements DoubleDataBuffer {

  public static DoubleLogicalDataBuffer map(ByteDataBuffer delegate, DoubleMapper doubleMapper) {
    return new DoubleLogicalDataBuffer(delegate, doubleMapper);
  }

  @Override
  public DoubleStream doubleStream() {
    return DoubleStream.iterate(0.0, d -> get((int)d)).limit(remaining());
  }

  @Override
  public DoubleDataBuffer get(double[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = doubleMapper.readDouble(physicalBuffer());
    }
    return this;
  }

  @Override
  public DoubleDataBuffer put(double[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      doubleMapper.writeDouble(physicalBuffer(), src[i]);
    }
    return this;
  }

  @Override
  public DoubleDataBuffer duplicate() {
    return new DoubleLogicalDataBuffer(physicalBuffer().duplicate(), doubleMapper);
  }

  private DoubleLogicalDataBuffer(ByteDataBuffer physicalBuffer, DoubleMapper doubleMapper) {
    super(physicalBuffer, doubleMapper);
    this.doubleMapper = doubleMapper;
  }

  private DoubleMapper doubleMapper;
}
