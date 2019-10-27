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

import java.util.stream.IntStream;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;

public class IntDataBufferSlice extends DataBufferSlice<Integer> implements IntDataBuffer {

    public IntDataBufferSlice(IntDataBuffer buffer) {
        super(buffer);
    }

    @Override
    public IntStream intStream() {
        // TODO
        throw new UnsupportedOperationException();
    }

    @Override
    public int getInt() {
        return delegate().getInt();
    }

    @Override
    public int getInt(long index) {
        return delegate().getInt(offset(index));
    }

    @Override
    public IntDataBufferSlice get(int[] dst, int offset, int length) {
        delegate().get(dst, offset, length);
        return this;
    }

    @Override
    public IntDataBufferSlice putInt(int value) {
        delegate().putInt(value);
        return this;
    }

    @Override
    public IntDataBufferSlice putInt(long index, int value) {
        delegate().putInt(offset(index), value);
        return this;
    }

    @Override
    public IntDataBufferSlice put(int[] src, int offset, int length) {
        delegate().put(src, offset, length);
        return this;
    }

    @Override
    public IntDataBufferSlice duplicate() {
        return new IntDataBufferSlice(delegate(), start(), end());
    }

    @Override
    public IntDataBufferSlice limit(long newLimit) {
        return (IntDataBufferSlice)super.limit(newLimit);
    }

    @Override
    public IntDataBufferSlice withLimit(long limit) {
        return (IntDataBufferSlice)super.withLimit(limit);
    }

    @Override
    public IntDataBufferSlice position(long newPosition) {
        return (IntDataBufferSlice)super.position(newPosition);
    }

    @Override
    public IntDataBufferSlice withPosition(long position) {
        return (IntDataBufferSlice)super.withPosition(position);
    }

    @Override
    public IntDataBufferSlice rewind() {
        return (IntDataBufferSlice)super.rewind();
    }

    @Override
    public IntDataBufferSlice put(Integer value) {
        return (IntDataBufferSlice)super.put(value);
    }

    @Override
    public IntDataBufferSlice put(long index, Integer value) {
        return (IntDataBufferSlice)super.put(index, value);
    }

    @Override
    public IntDataBufferSlice put(DataBuffer<Integer> src) {
        return (IntDataBufferSlice)super.put(src);
    }

    @Override
    protected IntDataBuffer delegate() {
        return (IntDataBuffer)super.delegate();
    }

    private IntDataBufferSlice(IntDataBuffer buffer, long start, long end) {
        super(buffer, start, end);
    }
}
