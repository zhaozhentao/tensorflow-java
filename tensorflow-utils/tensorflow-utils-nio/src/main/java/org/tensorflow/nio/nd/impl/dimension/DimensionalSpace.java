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

package org.tensorflow.nio.nd.impl.dimension;

import java.util.Arrays;
import org.tensorflow.nio.nd.IllegalRankException;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Index;

public class DimensionalSpace {

  public static DimensionalSpace create(Shape shape) {
    Dimension[] dimensions = new Dimension[shape.numDimensions()];

    // Start from the last dimension, where all elements are continuous
    for (int i = dimensions.length - 1, stride = 1; i >= 0; --i) {
      if (shape.size(i) == Shape.UNKNOWN_SIZE) {
        dimensions[i] = new UnknownDimension();
      } else {
        dimensions[i] = new Axis(shape.size(i), stride);
      }
      stride *= dimensions[i].numElements(); // FIXME what if unknown??
    }
    return new DimensionalSpace(dimensions, shape);
  }

  public DimensionalSpace mapTo(Index[] indices) {
    if (dimensions == null || indices.length > dimensions.length) {
      throw new ArrayIndexOutOfBoundsException();
    }
    Dimension[] newDimensions = Arrays.copyOf(dimensions, dimensions.length);
    for (int i = 0; i < indices.length; ++i) {
      newDimensions[i] = indices[i].apply(dimensions[i]);
    }
    return new DimensionalSpace(newDimensions);
  }

  public DimensionalSpace truncateFrom(int dimensionStart) {
    if (dimensionStart > dimensions.length) {
      throw new IndexOutOfBoundsException();
    }
    Dimension[] newDimensions = Arrays.copyOfRange(dimensions, dimensionStart, dimensions.length);
    return new DimensionalSpace(newDimensions);
  }

  public Shape shape() {
    if (shape == null) {
      shape = computeShape(dimensions);
    }
    return shape;
  }

  public int size() {
    return dimensions.length;
  }

  public Dimension get(int i) {
    return dimensions[i];
  }

  public long positionOf(long[] coords, boolean isValue) {
    if (coords.length > shape.numDimensions()) {
      throw new IndexOutOfBoundsException();
    }
    long position = 0L;
    int dimIdx = 0;
    for (long coord : coords) {
      position += dimensions[dimIdx++].positionOf(coord);
      // Fast-forward any remaining dimensions that are a single point
      while (dimIdx < dimensions.length && dimensions[dimIdx].isSinglePoint()) {
        position += dimensions[dimIdx++].position();
      }
    }
    if (isValue && dimIdx < shape.numDimensions()) {
      throw new IllegalRankException("Not a scalar value");
    }
    return position;
  }

  /** Succinct description of the shape meant for debugging. */
  @Override
  public String toString() {
    return Arrays.toString(dimensions);
  }

  private DimensionalSpace(Dimension[] dimensions) {
    this(dimensions, null);
  }

  private DimensionalSpace(Dimension[] dimensions, Shape shape) {
    this.dimensions = dimensions;
    this.shape = shape;
  }

  private final Dimension[] dimensions;
  private Shape shape;

  private static Shape computeShape(Dimension[] dimensions) {
    long[] shapeDimSizes = new long[dimensions.length];
    int numShapeDims = 0;
    for (Dimension dimension : dimensions) {
      if (!dimension.isSinglePoint()) {
        shapeDimSizes[numShapeDims++] = dimension.numElements();
      }
    }
    // TODO instead of truncating the shape dims, have a different constructor accepting a length
    return Shape.make(Arrays.copyOf(shapeDimSizes, numShapeDims));
  }
}
