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

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.nd.ByteNdArray;
import org.tensorflow.nio.nd.Shape;

public class ByteDenseNdArray extends AbstractDenseNdArray<Byte, ByteNdArray> implements ByteNdArray {

  public static ByteNdArray wrap(ByteDataBuffer buffer, Shape shape) {
    Validator.denseShape(shape);
    return new ByteDenseNdArray(buffer, shape);
  }

  @Override
  public ByteNdArray read(byte[] dst, int offset) {
    Validator.getArrayArgs(this, dst.length, offset);
    return read(DataBuffers.wrap(dst, false).position(offset));
  }

  @Override
  public ByteNdArray write(byte[] src, int offset) {
    Validator.putArrayArgs(this, src.length, offset);
    return write(DataBuffers.wrap(src, true).position(offset));
  }

  @Override
  public ByteNdArray setByte(byte value, long... indices) {
    buffer.putByte(position(indices, true), value);
    return this;
  }

  protected ByteDenseNdArray(ByteDataBuffer buffer, Shape shape) {
    super(shape);
    this.buffer = buffer;
  }

  @Override
  protected ByteDataBuffer buffer() {
    return buffer;
  }

  @Override
  protected ByteDenseNdArray allocateSlice(long position, Shape shape) {
    return new ByteDenseNdArray(buffer.withPosition(position).slice(), shape);
  }

  private ByteDataBuffer buffer;
}