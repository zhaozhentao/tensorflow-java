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

/** The possibly partially known shape of a tensor produced by an operation. */
public class UnknownShape implements Shape {

  @Override
  public UnknownShape mapTo(Index[] indices) {
    throw new ArrayIndexOutOfBoundsException();  // Unknown shapes cannot be mapped to indices
  }

  @Override
  public long size() {
    return UNKNOWN_SIZE;
  }

  @Override
  public long size(int i) {
    return UNKNOWN_SIZE;
  }

  @Override
  public int numDimensions() {
    return -1;
  }

  @Override
  public Dimension dimension(int i) {
    return null;
  }

  @Override
  public boolean hasUnknownDimension() {
    return true;
  }

  @Override
  public UnknownShape subshape(int dimensionStart) {
    return this;
  }

  @Override
  public long[] asArray() {
    return null;
  }

  @Override
  public boolean equals(Object obj) {
    return false;  // All unknown shapes are different
  }

  /** Succinct description of the shape meant for debugging. */
  @Override
  public String toString() {
    return "UNKNOWN";
  }

  UnknownShape() {}
}
