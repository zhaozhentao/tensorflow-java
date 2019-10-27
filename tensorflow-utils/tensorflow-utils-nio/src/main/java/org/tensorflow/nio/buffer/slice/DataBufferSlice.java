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
package org.tensorflow.nio.buffer.slice;

import java.util.stream.Stream;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

/**
 * A slice of a buffer that could be repositioned.
 *
 * @param <T> type of data
 */
public class DataBufferSlice<T> implements DataBuffer<T> {

  /**
   * Creates a new slice of a buffer, starting at {@code buffer.position()} and ending
   * at {@code buffer.limit()}.
   *
   * @param buffer buffer to slice
   * @param <T> type of data
   * @return buffer slice
   */
  public static <T> DataBufferSlice<T> create(DataBuffer<T> buffer) {
    return new DataBufferSlice<>(buffer, buffer.position(), buffer.limit());
  }

  /**
   * Returns the index in the origin buffer where this slice starts.
   */
  public long start() {
    return start;
  }

  /**
   * Returns the index in the origin buffer where this slice ends.
   */
  public long end() {
    return end;
  }

  /**
   * Move this slice to start at the provided index in its origin buffer.
   *
   * <p>The end of the slice remains the same. Both position and limit are reset to the start
   * and end boundaries of this slice respectively.
   *
   * @param start new start index
   */
  public void moveTo(long start) {
    this.start = start;
    delegate.position(start).limit(end);
  }

  /**
   * Move this slice to start and end at the provided indices in its origin buffer.
   *
   * <p>Both position and limit are reset to the new start and end boundaries of this slice
   * respectively.
   *
   * @param start new start index
   * @param end new end index
   */
  public void moveTo(long start, long end) {
    this.start = start;
    this.end = end;
    delegate.position(start).limit(end);
  }

  @Override
  public long capacity() {
    return end - start;
  }

  @Override
  public long limit() {
    return delegate.limit() - start;
  }

  @Override
  public DataBufferSlice<T> limit(long newLimit) {
    Validator.newLimit(this, newLimit);
    delegate.limit(newLimit + start);
    return this;
  }

  @Override
  public DataBufferSlice<T> withLimit(long limit) {
    duplicate().limit(limit);
    return this;
  }

  @Override
  public boolean hasRemaining() {
    return delegate.hasRemaining();
  }

  @Override
  public long remaining() {
    return delegate.remaining();
  }

  @Override
  public long position() {
    return delegate.position() - start;
  }

  @Override
  public DataBufferSlice<T> position(long newPosition) {
    Validator.newPosition(this, newPosition);
    delegate.position(newPosition + start);
    return this;
  }

  @Override
  public DataBufferSlice<T> withPosition(long position) {
    return duplicate().position(position);
  }

  @Override
  public DataBufferSlice<T> rewind() {
    delegate.position(start);
    return this;
  }

  @Override
  public boolean isReadOnly() {
    return delegate.isReadOnly();
  }

  @Override
  public T get() {
    return delegate.get();
  }

  @Override
  public T get(long index) {
    Validator.getArgs(this, index);
    return delegate.get(index + start);
  }

  @Override
  public Stream<T> stream() {
    // TODO
    throw new UnsupportedOperationException();
  }

  @Override
  public DataBufferSlice<T> put(T value) {
    delegate.put(value);
    return this;
  }

  @Override
  public DataBufferSlice<T> put(long index, T value) {
    Validator.putArgs(this, index);
    delegate.put(index + start, value);
    return this;
  }

  @Override
  public DataBufferSlice<T> put(DataBuffer<T> src) {
    delegate.put(src);
    return this;
  }

  @Override
  public DataBufferSlice<T> duplicate() {
    return new DataBufferSlice<>(delegate, start, end);
  }

  protected DataBuffer<T> delegate() {
    return delegate;
  }

  protected long offset(long index) {
    return index + start;
  }

  protected DataBufferSlice(DataBuffer<T> buffer, long start, long end) {
    this.start = start;
    this.end = end;
    this.delegate = buffer.duplicate();
  }

  private final DataBuffer<T> delegate;
  private long start;
  private long end;
}
