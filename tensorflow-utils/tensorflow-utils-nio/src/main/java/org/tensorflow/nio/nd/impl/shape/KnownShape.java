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

package org.tensorflow.nio.nd.impl.shape;

import java.util.Arrays;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Index;

/** The possibly partially known shape of a tensor produced by an operation. */
public class KnownShape implements Shape {

  @Override
  public KnownShape mapTo(Index[] indices) {
    if (dimensions == null || indices.length > dimensions.length) {
      throw new ArrayIndexOutOfBoundsException();
    }
    Dimension[] mappedDimensions = Arrays.copyOf(dimensions, dimensions.length);
    for (int i = 0; i < indices.length; ++i) {
      mappedDimensions[i] = indices[i].apply(dimensions[i]);
    }
    return new KnownShape(mappedDimensions);
  }

  @Override
  public long size() {
    return size;
  }

  @Override
  public long size(int i) {
    return dimensions[i].numElements();
  }

  @Override
  public int numDimensions() {
    return dimensions.length;
  }

  @Override
  public Dimension dimension(int i) {
    return dimensions[i];
  }

  @Override
  public boolean hasUnknownDimension() {
    return false;
  }

  @Override
  public Shape subshape(int dimensionStart) {
    return new KnownShape(Arrays.copyOfRange(dimensions, dimensionStart, dimensions.length));
  }

  @Override
  public long[] asArray() {
    return Arrays.stream(dimensions).mapToLong(Dimension::numElements).toArray();
  }

  @Override
  public int hashCode() {
    return Arrays.hashCode(dimensions);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    // Shapes are equivalent if all of their dimensions are equals
    if (obj instanceof KnownShape) {
      KnownShape otherShape = (KnownShape)obj;
      return Arrays.equals(dimensions, otherShape.dimensions);
    }
    return false;
  }

  /** Succinct description of the shape meant for debugging. */
  @Override
  public String toString() {
    return Arrays.toString(dimensions);
  }

  KnownShape(Dimension[] dimensions) {
    this.dimensions = dimensions;
    this.size = computeShapeSize(dimensions);
  }

  private final Dimension[] dimensions;
  private final long size;

  private static long computeShapeSize(Dimension[] dimensions) {
    long size = 1L;
    for (Dimension dimension: dimensions) {
      size *= dimension.numElements();
    }
    return size;
  }
}
