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
package org.tensorflow.nio.buffer.impl.large;

import java.nio.ReadOnlyBufferException;
import java.util.stream.DoubleStream;

import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.impl.single.DoubleJdkDataBuffer;

public final class DoubleLargeDataBuffer extends AbstractLargeDataBuffer<Double, DoubleDataBuffer> implements DoubleDataBuffer {

  public static long MAX_CAPACITY = DoubleJdkDataBuffer.MAX_CAPACITY * DoubleJdkDataBuffer.MAX_CAPACITY;

  public static DoubleDataBuffer allocate(long capacity) {
    if (capacity > MAX_CAPACITY) {
      throw new IllegalArgumentException("Capacity for a joined data buffer cannot exceeds " + MAX_CAPACITY + " bytes");
    }
    DoubleDataBuffer[] buffers = allocateBuffers(DoubleDataBuffer.class, capacity, DoubleJdkDataBuffer.MAX_CAPACITY, DoubleJdkDataBuffer::allocate);
    return new DoubleLargeDataBuffer(buffers, false);
  }

  public static DoubleDataBuffer join(DoubleDataBuffer... buffers) {
    boolean readOnly = Validator.joinBuffers(buffers);
    return new DoubleLargeDataBuffer(buffers, readOnly);
  }

  @Override
  public DoubleStream doubleStream() {
    DoubleStream stream = buffer(0).doubleStream();
    for (int i = 1; i < nbBuffers(); ++i) {
      stream = DoubleStream.concat(stream, buffer(i).doubleStream());
    }
    return stream;
  }

  @Override
  public double getDouble() {
    double value = currentBuffer().getDouble();
    onPositionIncrement();
    return value;
  }

  @Override
  public double getDouble(long index) {
    Validator.getArgs(this, index);
    int bufferIdx = bufferIndex(index);
    return buffer(bufferIdx).getDouble(indexInBuffer(bufferIdx, index));
  }

  @Override
  public DoubleDataBuffer get(double[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    copyArray(offset, length, (b, o, l) -> ((DoubleDataBuffer)b).get(dst, o, l));
    return this;
  }

  @Override
  public DoubleDataBuffer putDouble(double value) {
    Validator.put(this);
    currentBuffer().putDouble(value);
    onPositionIncrement();
    return this;
  }

  @Override
  public DoubleDataBuffer putDouble(long index, double value) {
    Validator.putArgs(this, index);
    int bufferIdx = bufferIndex(index);
    buffer(bufferIdx).putDouble(indexInBuffer(bufferIdx, index), value);
    return this;
  }

  @Override
  public DoubleDataBuffer put(double[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    copyArray(offset, length, (b, o, l) -> ((DoubleDataBuffer)b).put(src, o, l));
    return this;
  }

  @Override
  protected DoubleLargeDataBuffer instantiate(DoubleDataBuffer[] buffers, boolean readOnly) {
    return new DoubleLargeDataBuffer(buffers, readOnly);
  }

  private DoubleLargeDataBuffer(DoubleDataBuffer[] buffers, boolean readOnly) {
    super(buffers, readOnly);
  }
}
