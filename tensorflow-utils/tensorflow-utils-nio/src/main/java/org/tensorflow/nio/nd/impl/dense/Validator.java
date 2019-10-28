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
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.Shape;

final class Validator {

  static void denseShape(DataBuffer<?> buffer, Shape shape) {
    if (shape == null) {
      throw new IllegalArgumentException("Shape cannot be null");
    }
    if (shape.hasUnknownDimension()) {
      throw new IllegalArgumentException("Dense arrays cannot have unknown dimension(s)");
    }
    if (buffer.capacity() < shape.size()) {
      throw new IllegalArgumentException("Buffer capacity is smaller than the shape size");
    };
  }

  static void getArrayArgs(NdArray<?> ndArray, int arrayLength, int arrayOffset) {
    copyArrayArgs(arrayLength, arrayOffset);
    if (arrayLength - arrayOffset < ndArray.size()) {
      throw new BufferOverflowException();
    }
  }

  static void putArrayArgs(NdArray<?> ndArray, int arrayLength, int arrayOffset) {
    copyArrayArgs(arrayLength, arrayOffset);
    if (arrayLength - arrayOffset < ndArray.size()) {
      throw new BufferUnderflowException();
    }
  }

  static void copyNdArrayArgs(NdArray<?> ndArray, NdArray<?> otherNdArray) {
    if (!ndArray.shape().equals(otherNdArray.shape())) {
      throw new IllegalArgumentException("Can only copy to arrays of the same shape");
    }
  }

  private static void copyArrayArgs(int arrayLength, int arrayOffset) {
    if (arrayOffset < 0) {
      throw new IndexOutOfBoundsException("Offset must be non-negative");
    }
    if (arrayOffset > arrayLength) {
      throw new IndexOutOfBoundsException("Offset must be no larger than array length");
    }
  }

  private Validator() {}
}
