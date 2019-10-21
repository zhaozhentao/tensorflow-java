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
package org.tensorflow.nio.nd.impl;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.nd.ElementCursor;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dimension.DimensionalSpace;

@SuppressWarnings("unchecked")
public abstract class AbstractNdArray<T, U extends NdArray<T>> implements NdArray<T> {
  
  @Override
  public Shape shape() {
    return dimensions.shape();
  }

  @Override
  public ElementCursor<U> elements(int dimensionIdx) {
    if (dimensionIdx >= shape().numDimensions()) {
      throw new IllegalArgumentException("Cannot iterate elements in dimension '" + dimensionIdx +
          "' of array with shape " + shape());
    }
    return new DefaultElementCursor<>(dimensionIdx, (U)this);
  }

  @Override
  public ElementCursor<U> scalars() {
    return rank() == 0 ? new SingleElementCursor<>((U)this) : elements(shape().numDimensions() - 1);
  }

  @Override
  public U read(T[] dst) {
    return (U)read(DataBuffers.wrap(dst, false));
  }

  @Override
  public U read(T[] dst, int offset) {
    return (U)read(DataBuffers.wrap(dst, false).position(offset));
  }

  @Override public U write(T[] src) {
    return (U)write(DataBuffers.wrap(src, false));
  }

  @Override public U write(T[] src, int offset) {
    return (U)write(DataBuffers.wrap(src, false).position(offset));
  }

  public DimensionalSpace dimensions() {
    return dimensions;
  }

  protected AbstractNdArray(DimensionalSpace dimensions) {
    this.dimensions = dimensions;
  }

  protected void slowCopyTo(NdArray<T> array) {
    if (!shape().equals(array.shape())) {
      throw new IllegalArgumentException("Can only copy to arrays of the same shape");
    }
    scalars().forEachIdx((coords, e) -> array.setValue(e.getValue(), coords));
  }

  protected void slowRead(DataBuffer<T> buffer) {
    if (rank() == 0) {
      buffer.put(getValue());
    } else {
      scalars().forEach(e -> buffer.put(e.getValue()));
    }
  }

  protected void slowWrite(DataBuffer<T> buffer) {
    if (rank() == 0) {
      setValue(buffer.get());
    } else {
      scalars().forEach(e -> e.setValue(buffer.get()));
    }
  }

  private final DimensionalSpace dimensions;
}
