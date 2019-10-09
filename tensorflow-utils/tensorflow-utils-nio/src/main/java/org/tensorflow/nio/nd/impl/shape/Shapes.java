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

import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Index;

public final class Shapes {

  public static Shape unknown() {
    return new UnknownShape();
  }

  public static Shape scalar() {
    return new KnownShape(new Dimension[0]);
  }

  public static Shape make(long[] dimensionSizes) {
    Dimension[] dimensions = new Dimension[dimensionSizes.length];

    // Start from the last dimension, where all elements are continuous
    boolean partiallyKnown = false;
    for (int i = dimensionSizes.length - 1, positionStep = 1; i >= 0; --i) {
      if (dimensionSizes[i] == Shape.UNKNOWN_SIZE) {
        dimensions[i] = new UnknownDimension();
        partiallyKnown = true;
      } else {
        dimensions[i] = new Axis(dimensionSizes[i], positionStep);
      }
      positionStep *= dimensions[i].numElements();
    }
    return partiallyKnown ? new PartiallyKnownShape(dimensions) : new KnownShape(dimensions);
  }

  public static Dimension index(Dimension dim, Index index) {
    return new IndexedDimension((AbstractDimension)dim, index);
  }

  public static Dimension coordinate(long coord, Dimension dim) {
    return new Coordinate(coord, (AbstractDimension)dim);
  }

  private Shapes() {
  }
}
