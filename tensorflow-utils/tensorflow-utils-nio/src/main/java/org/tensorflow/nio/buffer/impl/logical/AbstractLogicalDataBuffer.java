package org.tensorflow.nio.buffer.impl.logical;

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.impl.AbstractDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

@SuppressWarnings("unchecked")
abstract class AbstractLogicalDataBuffer<T, B extends DataBuffer<T>> extends
    AbstractDataBuffer<T, B> {

  @Override
  public long capacity() {
    return physicalBuffer.capacity() / valueMapper.sizeInBytes();
  }

  @Override
  public long limit() {
    return physicalBuffer.limit() / valueMapper.sizeInBytes();
  }

  @Override
  public B limit(long newLimit) {
    physicalBuffer.limit(newLimit * valueMapper.sizeInBytes());
    return (B) this;
  }

  @Override
  public boolean hasRemaining() {
    return physicalBuffer.hasRemaining();
  }

  @Override
  public long remaining() {
    return physicalBuffer.remaining() / valueMapper.sizeInBytes();
  }

  @Override
  public long position() {
    return physicalBuffer.position() / valueMapper.sizeInBytes();
  }

  @Override
  public B position(long newPosition) {
    physicalBuffer.position(newPosition * valueMapper.sizeInBytes());
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
    return valueMapper.readValue(physicalBuffer());
  }

  @Override
  public T get(long index) {
    Validator.getArgs(this, index);
    // FIXME this duplicates the physical buffer on each call
    return valueMapper.readValue(physicalBuffer().withPosition(index * valueMapper.sizeInBytes()));
  }

  @Override
  public B put(T value) {
    valueMapper.writeValue(physicalBuffer(), value);
    return (B) this;
  }

  @Override
  public B put(long index, T value) {
    Validator.putArgs(this, index);
    // FIXME this duplicates the physical buffer on each call
    valueMapper.writeValue(physicalBuffer().withPosition(index * valueMapper.sizeInBytes()), value);
    return (B) this;
  }

  protected ByteDataBuffer physicalBuffer() {
    return physicalBuffer;
  }

  protected ValueMapper<T> valueMapper() {
    return valueMapper;
  }

  AbstractLogicalDataBuffer(ByteDataBuffer physicalBuffer, ValueMapper<T> valueMapper) {
    this.physicalBuffer = physicalBuffer;
    this.valueMapper = valueMapper;
  }

  private final ByteDataBuffer physicalBuffer;
  private final ValueMapper<T> valueMapper;
}
