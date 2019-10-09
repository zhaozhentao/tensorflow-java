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
import org.tensorflow.nio.nd.BooleanNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.ValueIterator;

public class BooleanDenseNdArray extends AbstractDenseNdArray<Boolean, BooleanNdArray> implements BooleanNdArray {

  public static BooleanNdArray wrap(BooleanDataBuffer buffer, Shape shape) {
    Validator.denseShape(shape);
    return new BooleanDenseNdArray(buffer, shape);
  }

  @Override
  public BooleanNdArray read(boolean[] dst, int offset) {
    Validator.getArrayArgs(this, dst.length, offset);
    if (isBulkCopyAvailable()) {
      BulkDataTransfer.create(this).execute((t, e) ->
          e.buffer().get(dst, offset + (int)t.totalCopied(), (int)t.bulkCopySize()).rewind()
      );
    } else {
      slowRead(dst, offset);
    }
    return this;
  }

  @Override
  public BooleanNdArray write(boolean[] src, int offset) {
    Validator.putArrayArgs(this, src.length, offset);
    if (isBulkCopyAvailable()) {
      BulkDataTransfer.create(this).execute((t, e) ->
          e.buffer().put(src, offset + (int)t.totalCopied(), (int)t.bulkCopySize()).rewind()
      );
    } else {
      slowWrite(src, offset);
    }
    return this;
  }

  @Override
  protected BooleanDataBuffer buffer() {
    return buffer;
  }

  @Override
  protected BooleanDenseNdArray allocateSlice(long position, Shape shape) {
    return new BooleanDenseNdArray(buffer.withPosition(position).slice(), shape);
  }

  protected BooleanDenseNdArray(BooleanDataBuffer buffer, Shape shape) {
    super(shape);
    this.buffer = buffer;
  }

  private void slowRead(boolean dst[], int offset) {
    int i = offset;
    for (Boolean v: values()) {
      dst[i++] = v;
    }
  }

  private void slowWrite(boolean src[], int offset) {
    int i = offset;
    for (ValueIterator<Boolean> iter = values().iterator(); iter.hasNext();) {
      iter.next(src[i++]);
    }
  }

  private BooleanDataBuffer buffer;
}