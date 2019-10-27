package org.tensorflow.nio.buffer.impl.logical;

import java.util.stream.Stream;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.converter.FloatDataConverter;
import org.tensorflow.nio.buffer.impl.Validator;

public class FloatLogicalDataBuffer extends AbstractLogicalDataBuffer<Float, FloatDataBuffer>
    implements FloatDataBuffer {

  public static FloatLogicalDataBuffer map(ByteDataBuffer delegate, FloatDataConverter floatMapper) {
    return new FloatLogicalDataBuffer(delegate, floatMapper);
  }

  @Override
  public float getFloat() {
    return converter.readFloat(physicalBuffer());
  }

  @Override
  public float getFloat(long index) {
    Validator.getArgs(this, index);
    return converter.readFloat(physicalBuffer().withPosition(index * converter.sizeInBytes()));
  }

  @Override
  public FloatDataBuffer get(float[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = converter.readFloat(physicalBuffer());
    }
    return this;
  }

  @Override
  public FloatDataBuffer putFloat(float value) {
    converter.writeFloat(physicalBuffer(), value);
    return this;
  }

  @Override
  public FloatDataBuffer putFloat(long index, float value) {
    Validator.putArgs(this, index);
    converter.writeFloat(physicalBuffer().withPosition(index * converter.sizeInBytes()), value);
    return this;
  }

  @Override
  public FloatDataBuffer put(float[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      converter.writeFloat(physicalBuffer(), src[i]);
    }
    return this;
  }

  @Override
  public Stream<Float> stream() {
    return Stream.iterate(0.0f, f -> get(f.intValue())).limit(remaining());
  }

  @Override
  public FloatDataBuffer duplicate() {
    return new FloatLogicalDataBuffer(physicalBuffer().duplicate(), converter);
  }

  private FloatLogicalDataBuffer(ByteDataBuffer physicalBuffer, FloatDataConverter converter) {
    super(physicalBuffer, converter);
    this.converter = converter;
  }

  private FloatDataConverter converter;
}
