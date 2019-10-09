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
 * An {@link NdArray} of booleans.
 */
public interface BooleanNdArray extends NdArray<Boolean> {

  /**
   * Reads the content of this N-dimensional array into the destination boolean array.
   *
   * <p>The size of the destination array must be equal or greater to the {@link #size()} of this array,
   * or an exception is thrown. After the copy, content of the both arrays can be altered
   * independently, without affecting each other.
   *
   * @param dst the destination array
   * @return this array
   * @throws java.nio.BufferOverflowException if the destination array cannot hold the content of this array
   */
  default BooleanNdArray read(boolean[] dst) {
    return read(dst, 0);
  }

  /**
   * Reads the content of this N-dimensional array into the destination boolean array.
   *
   * <p>{@code dst.length - offset} must be equal or greater to the {@link #size()} of this array,
   * or an exception is thrown. After the copy, content of the both arrays can be altered
   * independently, without affecting each other.
   *
   * @param dst the destination array
   * @param offset the index of the first boolean to write in the destination array
   * @return this array
   * @throws java.nio.BufferOverflowException if the destination array cannot hold the content of this array
   * @throws IndexOutOfBoundsException if offset is greater than dst length or is negative
   */
  BooleanNdArray read(boolean[] dst, int offset);

  /**
   * Writes the content of this N-dimensional array from the source boolean array.
   *
   * <p>The size of the source array must be equal or greater to the {@link #size()} of this array,
   * or an exception is thrown. After the copy, content of the both arrays can be altered
   * independently, without affecting each other.
   *
   * @param src the source array
   * @return this array
   * @throws java.nio.BufferUnderflowException if the size of the source array is less than the size of this array
   */
  default BooleanNdArray write(boolean[] src) {
    return write(src, 0);
  }

  /**
   * Writes the content of this N-dimensional array from the source boolean array.
   *
   * <p>{@code src.length - offset} must be equal or greater to the {@link #size()} of this array,
   * or an exception is thrown. After the copy, content of the both arrays can be altered
   * independently, without affecting each other.
   *
   * @param src the source array
   * @param offset the index of the first boolean to read from the source array
   * @return this array
   * @throws java.nio.BufferUnderflowException if the size of the source array is less than the size of this array
   * @throws IndexOutOfBoundsException if offset is greater than src length or is negative
   */
  BooleanNdArray write(boolean[] src, int offset);

  @Override
  BooleanNdArray at(long... indices);
  
  @Override
  BooleanNdArray slice(Index... indices);

  @Override
  Iterable<BooleanNdArray> elements();

  @Override
  BooleanNdArray set(Boolean value, long... indices);

  @Override
  BooleanNdArray copyTo(NdArray<Boolean> dst);

  @Override
  BooleanNdArray copyFrom(NdArray<Boolean> src);

  @Override
  BooleanNdArray read(DataBuffer<Boolean> dst);

  @Override
  BooleanNdArray write(DataBuffer<Boolean> src);

  @Override
  BooleanNdArray read(Boolean[] dst);

  @Override
  BooleanNdArray read(Boolean[] dst, int offset);

  @Override
  BooleanNdArray write(Boolean[] src);

  @Override
  BooleanNdArray write(Boolean[] src, int offset);
}
