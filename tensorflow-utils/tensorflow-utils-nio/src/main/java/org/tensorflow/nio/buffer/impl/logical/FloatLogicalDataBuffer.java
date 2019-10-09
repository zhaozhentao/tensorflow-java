package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.Stream;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

public class FloatLogicalDataBuffer extends AbstractLogicalDataBuffer<Float, FloatDataBuffer>
    implements FloatDataBuffer {

  public static FloatLogicalDataBuffer map(ByteDataBuffer delegate, FloatMapper floatMapper) {
    return new FloatLogicalDataBuffer(delegate, floatMapper);
  }

  @Override
  public FloatDataBuffer get(float[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = floatMapper.readFloat(physicalBuffer());
    }
    return this;
  }

  @Override
  public FloatDataBuffer put(float[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      floatMapper.writeFloat(physicalBuffer(), src[i]);
    }
    return this;
  }

  @Override
  public Stream<Float> stream() {
    return Stream.iterate(0.0f, f -> get(f.intValue())).limit(remaining());
  }

  @Override
  public FloatDataBuffer duplicate() {
    return new FloatLogicalDataBuffer(physicalBuffer().duplicate(), floatMapper);
  }

  private FloatLogicalDataBuffer(ByteDataBuffer physicalBuffer, FloatMapper floatMapper) {
    super(physicalBuffer, floatMapper);
    this.floatMapper = floatMapper;
  }

  private FloatMapper floatMapper;
}
