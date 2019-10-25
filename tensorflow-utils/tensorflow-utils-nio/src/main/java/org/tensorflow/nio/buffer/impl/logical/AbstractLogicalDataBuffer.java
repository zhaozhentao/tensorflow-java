package org.tensorflow.nio.buffer.impl.logical;

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.converter.DataConverter;
import org.tensorflow.nio.buffer.impl.AbstractDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

@SuppressWarnings("unchecked")
abstract class AbstractLogicalDataBuffer<T, B extends DataBuffer<T>> extends
    AbstractDataBuffer<T, B> {

  @Override
  public long capacity() {
    return physicalBuffer.capacity() / converter.sizeInBytes();
  }

  @Override
  public long limit() {
    return physicalBuffer.limit() / converter.sizeInBytes();
  }

  @Override
  public B limit(long newLimit) {
    physicalBuffer.limit(newLimit * converter.sizeInBytes());
    return (B) this;
  }

  @Override
  public boolean hasRemaining() {
    return physicalBuffer.hasRemaining();
  }

  @Override
  public long remaining() {
    return physicalBuffer.remaining() / converter.sizeInBytes();
  }

  @Override
  public long position() {
    return physicalBuffer.position() / converter.sizeInBytes();
  }

  @Override
  public B position(long newPosition) {
    physicalBuffer.position(newPosition * converter.sizeInBytes());
    return (B) this;
  }

  @Override
  public B rewind() {
    physicalBuffer.rewind();
    return (B) this;
  }

  @Override
  public boolean isReadOnly() {
    return physicalBuffer.isReadOnly();
  }

  @Override
  public T get() {
    return converter.readValue(physicalBuffer());
  }

  @Override
  public T get(long index) {
    Validator.getArgs(this, index);
    // FIXME this duplicates the physical buffer on each call
    return converter.readValue(physicalBuffer().withPosition(index * converter.sizeInBytes()));
  }

  @Override
  public B put(T value) {
    converter.writeValue(physicalBuffer(), value);
    return (B) this;
  }

  @Override
  public B put(long index, T value) {
    Validator.putArgs(this, index);
    // FIXME this duplicates the physical buffer on each call
    converter.writeValue(physicalBuffer().withPosition(index * converter.sizeInBytes()), value);
    return (B) this;
  }

  protected ByteDataBuffer physicalBuffer() {
    return physicalBuffer;
  }

  protected DataConverter<T> valueMapper() {
    return converter;
  }

  AbstractLogicalDataBuffer(ByteDataBuffer physicalBuffer, DataConverter<T> converter) {
    this.physicalBuffer = physicalBuffer;
    this.converter = converter;
  }

  private final ByteDataBuffer physicalBuffer;
  private final DataConverter<T> converter;
}
