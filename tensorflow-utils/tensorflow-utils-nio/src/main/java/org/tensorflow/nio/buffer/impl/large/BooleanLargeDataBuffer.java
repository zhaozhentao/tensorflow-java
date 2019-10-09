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

import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.impl.single.BitSetDataBuffer;

public final class BooleanLargeDataBuffer extends AbstractLargeDataBuffer<Boolean, BooleanDataBuffer> implements BooleanDataBuffer {

  public static long MAX_CAPACITY = BitSetDataBuffer.MAX_CAPACITY * BitSetDataBuffer.MAX_CAPACITY;

  public static BooleanDataBuffer allocate(long capacity) {
    if (capacity > MAX_CAPACITY) {
      throw new IllegalArgumentException("Capacity for a joined boolean data buffer cannot exceeds " + MAX_CAPACITY + " bytes");
    }
    BooleanDataBuffer[] buffers = allocateBuffers(BooleanDataBuffer.class, capacity, BitSetDataBuffer.MAX_CAPACITY, BitSetDataBuffer::allocate);
    return new BooleanLargeDataBuffer(buffers, false);
  }

  public static BooleanDataBuffer join(BooleanDataBuffer... buffers) {
    boolean readOnly = Validator.joinBuffers(buffers);
    return new BooleanLargeDataBuffer(buffers, readOnly);
  }

  @Override
  public BooleanDataBuffer get(boolean[] dst, int offset, int length) {
    Validator.getArrayArgs(this, dst.length, offset, length);
    copyArray(offset, length, (b, o, l) -> ((BooleanDataBuffer)b).get(dst, o, l));
    return this;
  }

  @Override
  public BooleanDataBuffer put(boolean[] src, int offset, int length) {
    Validator.putArrayArgs(this, src.length, offset, length);
    copyArray(offset, length, (b, o, l) -> ((BooleanDataBuffer)b).put(src, o, l));
    return this;
  }

  @Override
  protected BooleanLargeDataBuffer instantiate(BooleanDataBuffer[] buffers, boolean readOnly, long capacity, long limit, int currentBufferIndex) {
    return new BooleanLargeDataBuffer(buffers, readOnly, capacity, limit, currentBufferIndex);
  }

  private BooleanLargeDataBuffer(BooleanDataBuffer[] buffers, boolean readOnly) {
    super(buffers, readOnly);
  }

  private BooleanLargeDataBuffer(BooleanDataBuffer[] buffers, boolean readOnly, long capacity, long limit, int currentBufferIndex) {
    super(buffers, readOnly, capacity, limit, currentBufferIndex);
  }
}
