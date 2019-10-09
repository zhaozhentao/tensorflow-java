/*
 Copyright 2019 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =======================================================================
 */
package org.tensorflow.nio.buffer.impl.single;

import java.lang.reflect.Array;
import java.nio.BufferOverflowException;
import java.nio.BufferUnderflowException;
import java.nio.ReadOnlyBufferException;
import java.util.Arrays;
import java.util.stream.Stream;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.impl.AbstractBasicDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

public class ArrayDataBuffer<T> extends AbstractBasicDataBuffer<T, DataBuffer<T>> {

  public static long MAX_CAPACITY = Integer.MAX_VALUE - 2;
  
  public static <T> DataBuffer<T> allocate(Class<T> clazz, long capacity) {
    if (capacity < 0) {
      throw new IllegalArgumentException("Capacity must be non-negative");
    }
    if (capacity > MAX_CAPACITY) {
      throw new IllegalArgumentException("Size for an array-based data buffer cannot exceeds " + MAX_CAPACITY +
          " elements, use a LargeDataBuffer instead");
    }
    @SuppressWarnings("unchecked")
    T[] array = (T[])Array.newInstance(clazz, (int)capacity);
    return new ArrayDataBuffer<>(array, false);
  }

  public static <T> DataBuffer<T> wrap(T[] array, boolean readOnly) {
    return new ArrayDataBuffer<>(array, readOnly);
  }

  @Override
  public long capacity() {
    return values.length;
  }

  @Override
  public T get() {
    if (!hasRemaining()) {
      throw new BufferUnderflowException();
    }
    return values[(int)nextPosition()];
  }

  @Override
  public T get(long index) {
    Validator.getArgs(this, index);
    return values[(int)index];
  }

  @Override
  public Stream<T> stream() {
    return Arrays.stream(values);
  }

  @Override
  public DataBuffer<T> put(T value) {
    if (!hasRemaining()) {
      throw new BufferOverflowException();
    }
    if (isReadOnly()) {
      throw new ReadOnlyBufferException();
    }
    values[(int)nextPosition()] = value;
    return this;
  }

  @Override
  public DataBuffer<T> put(long index, T value) {
    Validator.putArgs(this, index);
    values[(int)index] = value;
    return this;
  }

  @Override
  public DataBuffer<T> put(DataBuffer<T> src) {
    Validator.putArgs(this, src);
    if (src instanceof ArrayDataBuffer) {
      ArrayDataBuffer<T> srcArrayBuffer = (ArrayDataBuffer<T>)src;
      int length = (int)src.remaining();
      System.arraycopy(srcArrayBuffer.values, (int) srcArrayBuffer.position(), values, (int) position(), length);
      srcArrayBuffer.movePosition(length);
      movePosition(length);
      return this;
    }
    return super.put(src);
  }

  @Override
  public DataBuffer<T> duplicate() {
    return new ArrayDataBuffer<>(values, isReadOnly(), (int) position(), (int) limit());
  }

  private ArrayDataBuffer(T[] values, boolean readOnly) {
    this(values, readOnly, 0, values.length);
  }

  private ArrayDataBuffer(T[] values, boolean readOnly, long position, long limit) {
    super(readOnly, position,  limit);
    this.values = values;  
  }
 
  private final T[] values;
}
