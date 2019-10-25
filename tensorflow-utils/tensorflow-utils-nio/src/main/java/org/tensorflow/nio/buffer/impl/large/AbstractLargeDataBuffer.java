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

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.function.BiConsumer;
import java.util.function.Function;
import java.util.stream.Stream;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.impl.AbstractDataBuffer;

@SuppressWarnings("unchecked")
abstract class AbstractLargeDataBuffer<T, B extends DataBuffer<T>> extends AbstractDataBuffer<T, B> {

  @Override
  public long capacity() {
    return bufferRanges[bufferRanges.length - 1].end;
  }

  @Override
  public long limit() {
    return limit;
  }

  @Override
  public B limit(long newLimit) {
    Validator.newLimit(this, newLimit);
    if (newLimit < position()) {
      currentBufferIdx = bufferIndex(newLimit);
    }
    resetBuffers(newLimit, B::limit);
    limit = newLimit;
    return (B)this;
  }

  @Override
  public boolean hasRemaining() {
    return currentBuffer().hasRemaining();
  }

  @Override
  public long remaining() {
    return limit - position();
  }

  @Override
  public long position() {
    long positionInBuffer = currentBuffer().position(); // validates current buffer
    return positionInBuffer + bufferRanges[currentBufferIdx].start;
  }

  @Override
  public B position(long newPosition) {
    Validator.newPosition(this, newPosition);
    resetBuffers(newPosition, B::position);
    currentBufferIdx = bufferIndex(newPosition);
    return (B)this;
  }

  @Override
  public B rewind() {
    currentBufferIdx = 0;
    for (B buffer: buffers) {
      buffer.rewind();
    }
    return (B)this;
  }

  @Override
  public boolean isReadOnly() {
    return readOnly;
  }

  @Override
  public T get() {
    T value = currentBuffer().get();
    onPositionIncrement();
    return value;
  }

  @Override
  public T get(long index) {
    Validator.getArgs(this, index);
    int bufferIdx = bufferIndex(index);
    return buffer(bufferIdx).get(indexInBuffer(bufferIdx, index));
  }

  @Override
  public Stream<T> stream() {
    Stream<T> stream = buffers[0].stream();
    for (int i = 1; i < buffers.length; ++i) {
      stream = Stream.concat(stream, buffers[i].stream());
    }
    return stream;
  }

  @Override
  public B put(T value) {
    Validator.put(this);
    currentBuffer().put(value);
    onPositionIncrement();
    return (B)this;
  }

  @Override
  public B put(long index, T value) {
    Validator.putArgs(this, index);
    int bufferIdx = bufferIndex(index);
    buffer(bufferIdx).put(indexInBuffer(bufferIdx, index), value);
    return (B)this;
  }

  @Override
  public B put(DataBuffer<T> src) {
    Validator.putArgs(this, src);
    long srcOriginalLimit = src.limit();
    long srcRemaining = src.remaining();
    while (srcRemaining > 0) {
      B buffer = currentBuffer();
      long length = buffer.remaining();
      if (length < srcRemaining) {
        buffer.put(src.limit(src.position() + length));
        srcRemaining -= length;
        ++currentBufferIdx;
      } else {
        buffer.put(src.limit(srcOriginalLimit));
        srcRemaining = 0;
      }
    }
    return (B)this;
  }

  @Override
  public B duplicate() {
    B[] duplicateBuffers = Arrays.stream(buffers).map(b -> (B)b.duplicate()).toArray(i -> Arrays.copyOf(buffers, i));
    AbstractLargeDataBuffer<T, B> duplicate = instantiate(duplicateBuffers, readOnly);
    duplicate.limit = limit;
    duplicate.currentBufferIdx = currentBufferIdx;
    return (B)duplicate;
  }

  abstract AbstractLargeDataBuffer<T, B> instantiate(B[] buffers, boolean readOnly);

  static <B extends DataBuffer<?>> B[] allocateBuffers(Class<B> bufferClazz, long capacity, long bufferMaxCapacity, Function<Long, B> allocator) {
    int nbMaxedBuffers = (int)(capacity / bufferMaxCapacity);
    long remaining = capacity % bufferMaxCapacity;
    B[] buffers = (B[]) Array.newInstance(bufferClazz, (remaining > 0 || nbMaxedBuffers == 0) ? nbMaxedBuffers + 1 : nbMaxedBuffers);
    int bufferIdx = 0;
    while (bufferIdx < nbMaxedBuffers) {
      buffers[bufferIdx++] = allocator.apply(bufferMaxCapacity);
    }
    if (bufferIdx < buffers.length) {
      buffers[bufferIdx] = allocator.apply(remaining);
    }
    return buffers;
  }

  AbstractLargeDataBuffer(B[] buffers, boolean readOnly) {
    if (buffers.length == 0) {
      throw new IllegalArgumentException("Buffers list cannot be empty");
    }
    this.buffers = buffers;
    this.bufferRanges = initBufferRanges(buffers);
    this.readOnly = readOnly;
    this.limit = capacity();
  }

  void onPositionIncrement() {
    if (currentBuffer().position() == currentBuffer().capacity() && currentBufferIdx < buffers.length - 1) {
      ++currentBufferIdx;
    }
  }

  int bufferIndex(long index) {
    int bufferIdx = 0;
    // Since we should have a relatively small number of buffers, we don't need to use binary search
    while (index >= bufferRanges[bufferIdx].end && bufferIdx < buffers.length - 1) {
      ++bufferIdx;
    }
    return bufferIdx;
  }

  B currentBuffer() {
    return buffers[currentBufferIdx];
  }

  B buffer(int bufferIdx) {
    return buffers[bufferIdx];
  }

  int nbBuffers() {
    return buffers.length;
  }

  int indexInBuffer(int bufferIdx, long index) {
    return (int)(index - bufferRanges[bufferIdx].start);
  }

  interface ArrayCopy<T> {
    void accept(DataBuffer<T> buf, int offset, int length);
  }

  void copyArray(int offset, int length, ArrayCopy<T> arrayCopy) {
    final int endIndex = offset + length;
    for (int index = offset; index < endIndex;) {
      int copyLength = endIndex - index;
      B buffer = currentBuffer();
      int bufferRemaining = (int)buffer.remaining();
      if (bufferRemaining < copyLength) {
        copyLength = bufferRemaining;
        ++currentBufferIdx;
      }
      arrayCopy.accept(buffer, index, copyLength);
      index += copyLength;
    }
  }

  private final B[] buffers;
  private final BufferRange[] bufferRanges;
  private final boolean readOnly;
  private long limit;
  private int currentBufferIdx;

  private void resetBuffers(long start, BiConsumer<B, Long> resetAction) {
    long remaining = start;
    for (B buffer: buffers) {
      long resetValue = Math.min(buffer.capacity(), remaining);
      resetAction.accept(buffer, resetValue);
      remaining -= resetValue;
    }
  }

  private BufferRange[] initBufferRanges(B[] buffers) {
    BufferRange[] ranges = new BufferRange[buffers.length];
    long start = 0;
    for (int i = 0; i < buffers.length; ++i) {
      ranges[i] = new BufferRange(start, start + buffers[i].capacity());
      start = ranges[i].end;
    }
    return ranges;
  }

  private static final class BufferRange {
    BufferRange(long start, long end) {
      this.start = start;
      this.end = end;
    }
    final long start;
    final long end;
  }
}
