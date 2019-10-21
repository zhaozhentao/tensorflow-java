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

import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.nd.DoubleNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dimension.DimensionalSpace;

public class DoubleDenseNdArray extends AbstractDenseNdArray<Double, DoubleNdArray> implements DoubleNdArray {

  public static DoubleNdArray create(DoubleDataBuffer buffer, Shape shape) {
    Validator.denseShape(shape);
    return new DoubleDenseNdArray(buffer, DimensionalSpace.create(shape));
  }

  @Override
  public double getDouble(long... indices) {
    return buffer().get(position(indices, true));
  }

  @Override
  public DoubleNdArray setDouble(double value, long... indices) {
    buffer().put(position(indices, true), value);
    return this;
  }

  @Override
  public DoubleNdArray read(double[] dst, int offset) {
    Validator.getArrayArgs(this, dst.length, offset);
    return read(DataBuffers.wrap(dst, false).position(offset));
  }

  @Override
  public DoubleNdArray write(double[] src, int offset) {
    Validator.putArrayArgs(this, src.length, offset);
    return write(DataBuffers.wrap(src, true).position(offset));
  }

  protected DoubleDenseNdArray(DoubleDataBuffer buffer, DimensionalSpace dimensions) {
    super(dimensions);
    this.buffer = buffer;
  }

  @Override
  protected DoubleDataBuffer buffer() {
    return buffer;
  }

  @Override
  protected DoubleDenseNdArray allocateSlice(long position, DimensionalSpace dimensions) {
    return new DoubleDenseNdArray(buffer.withPosition(position).slice(), dimensions);
  }

  private DoubleDataBuffer buffer;
}