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
package org.tensorflow.nio.nd.impl.dense;

import java.nio.BufferOverflowException;
import java.nio.BufferUnderflowException;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.nd.IllegalRankException;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.impl.AbstractNdArray;
import org.tensorflow.nio.nd.impl.dimension.DimensionalSpace;
import org.tensorflow.nio.nd.index.Index;

@SuppressWarnings("unchecked")
public abstract class AbstractDenseNdArray<T, U extends NdArray<T>> extends AbstractNdArray<T, U> {

  @Override
  public U slice(Index... indices) {
    DimensionalSpace sliceDimensions = dimensions().mapTo(indices);

    // Skip all leading dimensions that are a single point (i.e. a coordinate)
    long slicePosition = 0L;
    int i = 0;
    while (i < sliceDimensions.size() && sliceDimensions.get(i).isSinglePoint()) {
      slicePosition += sliceDimensions.get(i++).position();
    }
    if (i > 0) {
      sliceDimensions = sliceDimensions.truncateFrom(i);
    }

    return allocateSlice(slicePosition, sliceDimensions);
  }

  @Override
  public U get(long... indices) {
    DimensionalSpace sliceDimensions = dimensions().truncateFrom(indices.length);
    long slicePosition = position(indices, false);
    return allocateSlice(slicePosition, sliceDimensions);
  }

  @Override
  public T getValue(long... indices) {
    return buffer().get(position(indices, true));
  }

  @Override
  public U setValue(T value, long... indices) {
    buffer().put(position(indices, true), value);
    return (U)this;
  }

  @Override
  public U set(NdArray<T> src, long... coordinates) {
    src.copyTo((coordinates == null || coordinates.length == 0) ? this : get(coordinates));
    return (U)this;
  }

  @Override
  public U copyTo(NdArray<T> dst) {
    Validator.copyNdArrayArgs(this, dst);
    if (isContinuousInMemory()) {
      dst.write(buffer().duplicate());
    } else {
      super.slowCopyTo(dst);
    }
    return (U)this;
  }

  @Override
  public U read(DataBuffer<T> dst) {
    if (dst.remaining() < size()) {
      throw new BufferOverflowException();
    }
    if (isBulkDataTransferPossible()) {
      BulkDataTransfer.create(this).execute((t, e) ->
          dst.put(e.buffer().withLimit(t.bulkCopySize()))
      );
    } else {
      slowRead(dst);
    }
    return (U)this;
  }

  @Override
  public U write(DataBuffer<T> src) {
    if (src.remaining() < size()) {
      throw new BufferUnderflowException();
    }
    if (isBulkDataTransferPossible()) {
      BulkDataTransfer.create(this).execute((t, e) ->
          e.buffer().put(src.limit(src.position() + t.bulkCopySize())).rewind()
      );
    } else {
      slowWrite(src);
    }
    return (U)this;
  }

  AbstractDenseNdArray(DimensionalSpace dimensions) {
    super(dimensions);
  }

  protected abstract DataBuffer<T> buffer();

  protected abstract U allocateSlice(long position, DimensionalSpace dimensions);

  protected long position(long[] indices, boolean scalar) {
    if (indices.length > shape().numDimensions()) {
      throw new IndexOutOfBoundsException();
    }
    long position = 0L;
    int i = 0;
    for (; i < indices.length; ++i) {
      position += dimensions().get(i).positionOf(indices[i]);
    }
    while (i < dimensions().size() && dimensions().get(i).isSinglePoint()) {
      position += dimensions().get(i++).position();
    }
    if (scalar && i < shape().numDimensions()) {
      throw new IllegalRankException("Not a scalar value");
    }
    return position;
  }

  /**
   * Check if we copy this array data in bulk. Bulk copy is only possible for array of 1-dimension or more and that
   * the last dimension is not segmented (therefore linear in memory).
   *
   * @return true if bulk copy is possible
   */
  boolean isBulkDataTransferPossible() {
    return dimensions().size() > 0 && !dimensions().get(shape().numDimensions() - 1).isSegmented();
  }

  private boolean isContinuousInMemory() {
    for (int i = 0; i < shape().numDimensions(); ++i) {
      if (dimensions().get(i).isSegmented()) {
        return false;
      }
    }
    return true;
  }
}
