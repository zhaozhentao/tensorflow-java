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
public class PartiallyKnownShape extends KnownShape {

  @Override
  public long size() {
    return UNKNOWN_SIZE;
  }

  @Override
  public boolean hasUnknownDimension() {
    return true;
  }

  @Override
  public Shape subshape(int dimensionStart) {
    Dimension[] subDimensions = new Dimension[numDimensions() - dimensionStart];
    boolean partiallyKnown = false;
    for (int i = 0; i < subDimensions.length; ++i) {
      Dimension dim = dimension(i + dimensionStart);
      if (dim.numElements() == Shape.UNKNOWN_SIZE) {
        partiallyKnown = true;
      }
      subDimensions[i] = dim;
    }
    return partiallyKnown ? new PartiallyKnownShape(subDimensions) : new KnownShape(subDimensions);
  }

  @Override
  public boolean equals(Object obj) {
    return false;  // All partially known shapes are different
  }

  PartiallyKnownShape(Dimension[] dimensions) {
    super(dimensions);
  }
}
