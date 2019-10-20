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
import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.impl.AbstractBasicDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

public class BooleanArrayDataBuffer extends AbstractBasicDataBuffer<Boolean, BooleanDataBuffer> implements BooleanDataBuffer {

  public static BooleanDataBuffer wrap(boolean[] array, boolean readOnly) {
    return new BooleanArrayDataBuffer(array, readOnly);
  }

  @Override
  public long capacity() {
    return values.length;
  }

  @Override
  public Boolean get() {
    if (!hasRemaining()) {
      throw new BufferUnderflowException();
    }
    return values[(int)nextPosition()];
  }

  @Override
  public BooleanDataBuffer get(boolean[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    System.arraycopy(values, (int)position(), dst, offset, length);
    return this;
  }

  @Override
  public Boolean get(long index) {
    Validator.getArgs(this, index);
    return values[(int)index];
  }

  @Override
  public Stream<Boolean> stream() {
    throw new UnsupportedOperationException(); // TODO!
  }

  @Override
  public BooleanDataBuffer put(Boolean value) {
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
  public BooleanDataBuffer put(long index, Boolean value) {
    Validator.putArgs(this, index);
    values[(int)index] = value;
    return this;
  }

  @Override
  public BooleanDataBuffer put(DataBuffer<Boolean> src) {
    Validator.putArgs(this, src);
    if (src instanceof BooleanArrayDataBuffer) {
      BooleanArrayDataBuffer srcArrayBuffer = (BooleanArrayDataBuffer)src;
      int length = (int)src.remaining();
      System.arraycopy(srcArrayBuffer.values, (int) srcArrayBuffer.position(), values, (int) position(), length);
      srcArrayBuffer.movePosition(length);
      movePosition(length);
      return this;
    }
    return super.put(src);
  }

  @Override
  public BooleanDataBuffer put(boolean[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    System.arraycopy(src, offset, values, (int)position(), length);
    return this;
  }

  @Override
  public BooleanDataBuffer duplicate() {
    return new BooleanArrayDataBuffer(values, isReadOnly(), (int) position(), (int) limit());
  }

  private BooleanArrayDataBuffer(boolean[] values, boolean readOnly) {
    this(values, readOnly, 0, values.length);
  }

  private BooleanArrayDataBuffer(boolean[] values, boolean readOnly, long position, long limit) {
    super(readOnly, position,  limit);
    this.values = values;  
  }
 
  private final boolean[] values;
}
