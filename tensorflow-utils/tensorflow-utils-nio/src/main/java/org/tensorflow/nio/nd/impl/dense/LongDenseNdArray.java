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

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.nd.LongNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dimension.DimensionalSpace;

public class LongDenseNdArray extends AbstractDenseNdArray<Long, LongNdArray>
    implements LongNdArray {

  public static LongNdArray create(LongDataBuffer buffer, Shape shape) {
    Validator.denseShape(buffer, shape);
    return new LongDenseNdArray(buffer, shape);
  }

  @Override
  public long getLong(long... indices) {
    return buffer().getLong(positionOf(indices, true));
  }

  @Override
  public LongNdArray setLong(long value, long... indices) {
    buffer().putLong(positionOf(indices, true), value);
    return this;
  }

  @Override
  public LongNdArray read(long[] dst, int offset) {
    Validator.getArrayArgs(this, dst.length, offset);
    return read(DataBuffers.wrap(dst, false).position(offset));
  }

  @Override
  public LongNdArray write(long[] src, int offset) {
    Validator.putArrayArgs(this, src.length, offset);
    return write(DataBuffers.wrap(src, true).position(offset));
  }

  protected LongDenseNdArray(LongDataBuffer buffer, Shape shape) {
    this(buffer, DimensionalSpace.create(shape));
  }

  @Override
  @SuppressWarnings("unchecked")
  protected LongDataBuffer buffer() {
    return super.buffer();
  }

  @Override
  LongDenseNdArray allocate(DataBuffer<Long> buffer, DimensionalSpace dimensions) {
    return new LongDenseNdArray((LongDataBuffer)buffer, dimensions);
  }

  private LongDenseNdArray(LongDataBuffer buffer, DimensionalSpace dimensions) {
    super(buffer, dimensions);
  }
}