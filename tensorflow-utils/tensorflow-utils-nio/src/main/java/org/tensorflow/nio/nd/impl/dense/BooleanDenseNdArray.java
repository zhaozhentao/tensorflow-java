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

import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.nd.BooleanNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dimension.DimensionalSpace;

public class BooleanDenseNdArray extends AbstractDenseNdArray<Boolean, BooleanNdArray>
    implements BooleanNdArray {

  public static BooleanNdArray create(BooleanDataBuffer buffer, Shape shape) {
    Validator.denseShape(shape);
    return new BooleanDenseNdArray(buffer, DimensionalSpace.create(shape));
  }

  @Override
  public boolean getBoolean(long... indices) {
    return buffer().getBoolean(positionOf(indices, true));
  }

  @Override
  public BooleanNdArray setBoolean(boolean value, long... indices) {
    buffer().putBoolean(positionOf(indices, true), value);
    return this;
  }

  @Override
  public BooleanNdArray read(boolean[] dst, int offset) {
    Validator.getArrayArgs(this, dst.length, offset);
    return read(DataBuffers.wrap(dst, false).position(offset));
  }

  @Override
  public BooleanNdArray write(boolean[] src, int offset) {
    Validator.putArrayArgs(this, src.length, offset);
    return write(DataBuffers.wrap(src, true).position(offset));
  }

  protected BooleanDenseNdArray(BooleanDataBuffer buffer, DimensionalSpace dimensions) {
    super(buffer, dimensions);
  }

  @Override
  protected BooleanDataBuffer buffer() {
    return super.buffer();
  }

  @Override
  BooleanDenseNdArray allocate(DataBuffer<Boolean> buffer, DimensionalSpace dimensions) {
    return new BooleanDenseNdArray((BooleanDataBuffer)buffer, dimensions);
  }
}