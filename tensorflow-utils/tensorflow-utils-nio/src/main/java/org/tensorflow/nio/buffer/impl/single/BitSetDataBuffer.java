package org.tensorflow.nio.buffer.impl.single;

import java.nio.BufferOverflowException;
import java.nio.BufferUnderflowException;
import java.util.BitSet;
import java.util.stream.Stream;
import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.impl.AbstractBasicDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

public class BitSetDataBuffer extends
    AbstractBasicDataBuffer<Boolean, BooleanDataBuffer> implements BooleanDataBuffer  {

  public static long MAX_CAPACITY = Integer.MAX_VALUE - 2;

  public static BooleanDataBuffer allocate(long capacity) {
    if (capacity < 0) {
      throw new IllegalArgumentException("Capacity must be non-negative");
    }
    if (capacity > MAX_CAPACITY) {
      throw new IllegalArgumentException("Size for an bit-set data buffer cannot exceeds " + MAX_CAPACITY +
          " elements, use a LargeDataBuffer instead");
    }
    return new BitSetDataBuffer(new BitSet((int)capacity), capacity, false);
  }

  @Override
  public BooleanDataBuffer get(boolean[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      dst[i] = bitSet.get((int)nextPosition());
    }
    return this;
  }

  @Override
  public BooleanDataBuffer put(boolean[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    for (int i = offset; i < offset + length; ++i) {
      bitSet.set((int)nextPosition(), src[i]);
    }
    return this;
  }

  @Override
  public long capacity() {
    return capacity;
  }

  @Override
  public Boolean get() {
    if (position() >= capacity()) {
      throw new BufferUnderflowException();
    }
    return bitSet.get((int)nextPosition());
  }

  @Override
  public Boolean get(long index) {
    Validator.getArgs(this, index);
    return bitSet.get((int)index);
  }

  @Override
  public Stream<Boolean> stream() {
    throw new UnsupportedOperationException("BooleanDataBuffer does not support value streaming at the moment");
  }

  @Override
  public BooleanDataBuffer put(Boolean value) {
    if (position() >= capacity()) {
      throw new BufferOverflowException();
    }
    bitSet.set((int)nextPosition(), value);
    return this;
  }

  @Override
  public BooleanDataBuffer put(long index, Boolean value) {
    Validator.putArgs(this, index);
    bitSet.set((int)index, value);
    return this;
  }

  @Override
  public BooleanDataBuffer duplicate() {
    return new BitSetDataBuffer(bitSet, capacity, isReadOnly(), position(), limit());
  }

  private BitSetDataBuffer(BitSet bitSet, long capacity, boolean readOnly) {
    this(bitSet, capacity, readOnly, 0, capacity);
  }

  private BitSetDataBuffer(BitSet bitSet, long capacity, boolean readOnly, long position, long limit) {
    super(readOnly, position, limit);
    this.capacity = capacity;
    this.bitSet = bitSet;
  }

  private final BitSet bitSet;
  private final long capacity;
}
