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
package org.tensorflow.nio.buffer.slice;

import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;

/**
 * A slice of a boolean buffer that could be repositioned.
 */
public class BooleanDataBufferSlice extends DataBufferSlice<Boolean> implements BooleanDataBuffer {

    /**
     * Creates a new slice of a boolean buffer, starting at {@code buffer.position()} and ending
     * at {@code buffer.limit()}.
     *
     * @param buffer buffer to slice
     * @return buffer slice
     */
    public static BooleanDataBufferSlice create(BooleanDataBuffer buffer) {
        return new BooleanDataBufferSlice(buffer, buffer.position(), buffer.limit());
    }

    @Override
    public boolean getBoolean() {
        return delegate().getBoolean();
    }

    @Override
    public boolean getBoolean(long index) {
        return delegate().getBoolean(offset(index));
    }

    @Override
    public BooleanDataBufferSlice get(boolean[] dst, int offset, int length) {
        delegate().get(dst, offset, length);
        return this;
    }

    @Override
    public BooleanDataBufferSlice putBoolean(boolean value) {
        delegate().putBoolean(value);
        return this;
    }

    @Override
    public BooleanDataBufferSlice putBoolean(long index, boolean value) {
        delegate().putBoolean(offset(index), value);
        return this;
    }

    @Override
    public BooleanDataBufferSlice put(boolean[] src, int offset, int length) {
        delegate().put(src, offset, length);
        return this;
    }

    @Override
    public BooleanDataBufferSlice duplicate() {
        return new BooleanDataBufferSlice(delegate().duplicate(), start(), end());
    }

    @Override
    public BooleanDataBufferSlice limit(long newLimit) {
        return (BooleanDataBufferSlice)super.limit(newLimit);
    }

    @Override
    public BooleanDataBufferSlice withLimit(long limit) {
        return (BooleanDataBufferSlice)super.withLimit(limit);
    }

    @Override
    public BooleanDataBufferSlice position(long newPosition) {
        return (BooleanDataBufferSlice)super.position(newPosition);
    }

    @Override
    public BooleanDataBufferSlice withPosition(long position) {
        return (BooleanDataBufferSlice)super.withPosition(position);
    }

    @Override
    public BooleanDataBufferSlice rewind() {
        return (BooleanDataBufferSlice)super.rewind();
    }

    @Override
    public BooleanDataBufferSlice put(Boolean value) {
        return (BooleanDataBufferSlice)super.put(value);
    }

    @Override
    public BooleanDataBufferSlice put(long index, Boolean value) {
        return (BooleanDataBufferSlice)super.put(index, value);
    }

    @Override
    public BooleanDataBufferSlice put(DataBuffer<Boolean> src) {
        return (BooleanDataBufferSlice)super.put(src);
    }

    @Override
    protected BooleanDataBuffer delegate() {
        return (BooleanDataBuffer)super.delegate();
    }

    private BooleanDataBufferSlice(BooleanDataBuffer buffer, long start, long end) {
        super(buffer, start, end);
    }
}
