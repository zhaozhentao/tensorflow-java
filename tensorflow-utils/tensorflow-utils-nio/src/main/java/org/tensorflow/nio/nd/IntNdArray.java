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

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.nd.index.Index;

/**
 * An {@link NdArray} of integers.
 */
public interface IntNdArray extends NdArray<Integer> {

  /**
   * Returns the integer value of the scalar found at the given coordinates.
   *
   * <p>To access the scalar element, the number of indices provided must be equal to the number
   * of dimensions of this array (i.e. its rank). For example:
   * <pre>{@code
   *  IntNdArray matrix = NdArrays.ofInts(shape(2, 2));  // matrix rank = 2
   *  matrix.get(0, 1);  // succeeds, returns 0
   *  matrix.get(0);  // throws IllegalRankException
   *
   *  IntNdArray scalar = matrix.at(0, 1);  // scalar rank = 0
   *  scalar.get();  // succeeds, returns 0
   * }</pre>
   *
   * @param indices coordinates of the scalar to resolve
   * @return value of that scalar
   * @throws IndexOutOfBoundsException if some indices are outside the limits of their respective dimension
   * @throws IllegalRankException if number of indices is not sufficient to access a scalar element
   */
  int get(long... indices);

  /**
   * Reads the content of this N-dimensional array into the destination int array.
   *
   * <p>The size of the destination array must be equal or greater to the {@link #size()} of this array,
   * or an exception is thrown. After the copy, content of the both arrays can be altered
   * independently, without affecting each other.
   *
   * @param dst the destination array
   * @return this array
   * @throws java.nio.BufferOverflowException if the destination array cannot hold the content of this array
   */
  default IntNdArray read(int[] dst) {
    return read(dst, 0);
  }

  /**
   * Reads the content of this N-dimensional array into the destination int array.
   *
   * <p>{@code dst.length - offset} must be equal or greater to the {@link #size()} of this array,
   * or an exception is thrown. After the copy, content of the both arrays can be altered
   * independently, without affecting each other.
   *
   * @param dst the destination array
   * @param offset the index of the first integer to write in the destination array
   * @return this array
   * @throws java.nio.BufferOverflowException if the destination array cannot hold the content of this array
   * @throws IndexOutOfBoundsException if offset is greater than dst length or is negative
   */
  IntNdArray read(int[] dst, int offset);

  /**
   * Assigns the integer value of the scalar found at the given coordinates.
   *
   * <p>To access the scalar element, the number of indices provided must be equal to the number
   * of dimensions of this array (i.e. its rank). For example:
   * <pre>{@code
   *  IntNdArray matrix = NdArrays.ofInts(shape(2, 2));  // matrix rank = 2
   *  matrix.set(10, 0, 1);  // succeeds
   *  matrix.set(10, 0);  // throws IllegalRankException
   *
   *  IntNdArray scalar = matrix.at(0, 1);  // scalar rank = 0
   *  scalar.set(10);  // succeeds
   * }</pre>
   *
   * @param indices coordinates of the scalar to assign
   * @return this array
   * @throws IndexOutOfBoundsException if some indices are outside the limits of their respective dimension
   * @throws IllegalRankException if number of indices is not sufficient to access a scalar element
   */
  IntNdArray set(int value, long... indices);

  /**
   * Writes the content of this N-dimensional array from the source int array.
   *
   * <p>The size of the source array must be equal or greater to the {@link #size()} of this array,
   * or an exception is thrown. After the copy, content of the both arrays can be altered
   * independently, without affecting each other.
   *
   * @param src the source array
   * @return this array
   * @throws java.nio.BufferUnderflowException if the size of the source array is less than the size of this array
   */
  default IntNdArray write(int[] src) {
    return write(src, 0);
  }

  /**
   * Writes the content of this N-dimensional array from the source int array.
   *
   * <p>{@code src.length - offset} must be equal or greater to the {@link #size()} of this array,
   * or an exception is thrown. After the copy, content of the both arrays can be altered
   * independently, without affecting each other.
   *
   * @param src the source array
   * @param offset the index of the first integer to read from the source array
   * @return this array
   * @throws java.nio.BufferUnderflowException if the size of the source array is less than the size of this array
   * @throws IndexOutOfBoundsException if offset is greater than src length or is negative
   */
  IntNdArray write(int[] src, int offset);

  @Override
  IntNdArray at(long... indices);
  
  @Override
  IntNdArray slice(Index... indices);

  @Override
  Iterable<IntNdArray> elements();

  @Override
  IntNdArray setValue(Integer value, long... indices);

  @Override
  IntNdArray copyTo(NdArray<Integer> dst);

  @Override
  IntNdArray copyFrom(NdArray<Integer> src);

  @Override
  IntNdArray read(DataBuffer<Integer> dst);

  @Override
  IntNdArray write(DataBuffer<Integer> src);

  @Override
  IntNdArray read(Integer[] dst);

  @Override
  IntNdArray read(Integer[] dst, int offset);

  @Override
  IntNdArray write(Integer[] src);

  @Override
  IntNdArray write(Integer[] src, int offset);
}
