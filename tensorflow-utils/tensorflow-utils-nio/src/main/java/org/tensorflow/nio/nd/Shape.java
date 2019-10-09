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

package org.tensorflow.nio.nd;

import org.tensorflow.nio.nd.impl.shape.Dimension;
import org.tensorflow.nio.nd.impl.shape.Shapes;
import org.tensorflow.nio.nd.index.Index;

/** The possibly partially known shape of a tensor produced by an operation. */
public interface Shape {

  long UNKNOWN_SIZE = -1L;

  /** Create a Shape representing an unknown number of dimensions. */
  static Shape unknown() {
    return Shapes.unknown();
  }

  /** Create a Shape representing a scalar value. */
  static Shape scalar() {
    return Shapes.scalar();
  }

  /**
   * Create a Shape representing an N-dimensional value.
   *
   * <p>Creates a Shape representing an N-dimensional value (N being at least 1), with the provided
   * size for each dimension. A -1 indicates that the size of the corresponding dimension is
   * unknown. For example:
   *
   * <pre>{@code
   * // A 2-element vector.
   * Shape vector = Shape.create(2);
   *
   * // A 2x3 matrix.
   * Shape matrix = Shape.create(2, 3);
   *
   * // A matrix with 4 columns but an unknown number of rows.
   * // This is typically used to indicate the shape of tensors that represent
   * // a variable-sized batch of values. The Shape below might represent a
   * // variable-sized batch of 4-element vectors.
   * Shape batch = Shape.create(-1, 4);
   * }</pre>
   */
  static Shape make(long... dimensionSizes) {
    if (dimensionSizes == null) {
      return Shapes.scalar();
    }
    return Shapes.make(dimensionSizes);
  }

  Shape mapTo(Index[] indices);

  long size();

  long size(int i);

  /**
   * Number of dimensions represented by this shape.
   *
   * @return -1 if the number of dimensions is unknown, 0 if the shape represents a scalar, 1 for a
   *     vector, 2 for a matrix etc.
   */
  int numDimensions();

  Dimension dimension(int i);

  boolean hasUnknownDimension();

  Shape subshape(int dimensionStart);

  long[] asArray();
}
