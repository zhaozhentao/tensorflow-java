package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.DoubleStream;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.converter.DoubleDataConverter;
import org.tensorflow.nio.buffer.impl.Validator;

public class DoubleLogicalDataBuffer extends AbstractLogicalDataBuffer<Double, DoubleDataBuffer>
    implements DoubleDataBuffer {

  public static DoubleLogicalDataBuffer map(ByteDataBuffer delegate, DoubleDataConverter doubleMapper) {
    return new DoubleLogicalDataBuffer(delegate, doubleMapper);
  }

  @Override
  public DoubleStream doubleStream() {
    return DoubleStream.iterate(0.0, d -> get((int)d)).limit(remaining());
  }

  @Override
  public double getDouble() {
    return converter.readDouble(physicalBuffer());
  }

  @Override
  public double getDouble(long index) {
    Validator.getArgs(this, index);
    return converter.readDouble(physicalBuffer().withPosition(index * converter.sizeInBytes()));
  }

  @Override
  public DoubleDataBuffer get(double[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = converter.readDouble(physicalBuffer());
    }
    return this;
  }

  @Override
  public DoubleDataBuffer putDouble(double value) {
    converter.writeDouble(physicalBuffer(), value);
    return this;
  }

  @Override
  public DoubleDataBuffer putDouble(long index, double value) {
    Validator.putArgs(this, index);
    converter.writeDouble(physicalBuffer().withPosition(index * converter.sizeInBytes()), value);
    return this;
  }

  @Override
  public DoubleDataBuffer put(double[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      converter.writeDouble(physicalBuffer(), src[i]);
    }
    return this;
  }

  @Override
  public DoubleDataBuffer duplicate() {
    return new DoubleLogicalDataBuffer(physicalBuffer().duplicate(), converter);
  }

  private DoubleLogicalDataBuffer(ByteDataBuffer physicalBuffer, DoubleDataConverter converter) {
    super(physicalBuffer, converter);
    this.converter = converter;
  }

  private DoubleDataConverter converter;
}
