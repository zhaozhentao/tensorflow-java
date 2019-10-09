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
package org.tensorflow.nio.buffer.impl;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.impl.AbstractDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;

@SuppressWarnings("unchecked")
public abstract class AbstractBasicDataBuffer<T, B extends DataBuffer<T>> extends AbstractDataBuffer<T, B> {

  @Override
  public long limit() {
    return limit;
  }

  @Override
  public B limit(long newLimit) {
    Validator.newLimit(this, newLimit);
    limit = newLimit;
    if (position > limit) {
      position = limit;
    }
    return (B)this;
  }

  @Override
  public boolean hasRemaining() {
    return position < limit;
  }

  @Override
  public long remaining() {
    return limit - position;
  }

  @Override
  public long position() {
    return position;
  }

  @Override
  public B position(long newPosition) {
    Validator.newPosition(this, newPosition);
    position = newPosition;
    return (B)this;
  }

  @Override
  public B rewind() {
    position = 0;
    return (B)this;
  }

  @Override
  public boolean isReadOnly() {
    return readOnly;
  }

  protected AbstractBasicDataBuffer(boolean readOnly, long position, long limit) {
    this.readOnly = readOnly;
    this.position = position;
    this.limit = limit;
  }

  protected long nextPosition() {
    return position++;
  }

  protected void movePosition(long step) {
    position += step;
  }

  private final boolean readOnly;
  private long position;
  private long limit;
}
