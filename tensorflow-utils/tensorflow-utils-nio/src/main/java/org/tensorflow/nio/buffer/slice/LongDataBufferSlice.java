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

import java.util.stream.LongStream;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;

public class LongDataBufferSlice extends DataBufferSlice<Long> implements LongDataBuffer {

    public LongDataBufferSlice(LongDataBuffer buffer) {
        super(buffer);
    }

    @Override
    public LongStream longStream() {
        // TODO
        throw new UnsupportedOperationException();
    }

    @Override
    public long getLong() {
        return delegate().getLong();
    }

    @Override
    public long getLong(long index) {
        return delegate().getLong(offset(index));
    }

    @Override
    public LongDataBufferSlice get(long[] dst, int offset, int length) {
        delegate().get(dst, offset, length);
        return this;
    }

    @Override
    public LongDataBufferSlice putLong(long value) {
        delegate().putLong(value);
        return this;
    }

    @Override
    public LongDataBufferSlice putLong(long index, long value) {
        delegate().putLong(offset(index), value);
        return this;
    }

    @Override
    public LongDataBufferSlice put(long[] src, int offset, int length) {
        delegate().put(src, offset, length);
        return this;
    }

    @Override
    public LongDataBufferSlice duplicate() {
        return new LongDataBufferSlice(delegate(), start(), end());
    }

    @Override
    public LongDataBufferSlice limit(long newLimit) {
        return (LongDataBufferSlice)super.limit(newLimit);
    }

    @Override
    public LongDataBufferSlice withLimit(long limit) {
        return (LongDataBufferSlice)super.withLimit(limit);
    }

    @Override
    public LongDataBufferSlice position(long newPosition) {
        return (LongDataBufferSlice)super.position(newPosition);
    }

    @Override
    public LongDataBufferSlice withPosition(long position) {
        return (LongDataBufferSlice)super.withPosition(position);
    }

    @Override
    public LongDataBufferSlice rewind() {
        return (LongDataBufferSlice)super.rewind();
    }

    @Override
    public LongDataBufferSlice put(Long value) {
        return (LongDataBufferSlice)super.put(value);
    }

    @Override
    public LongDataBufferSlice put(long index, Long value) {
        return (LongDataBufferSlice)super.put(index, value);
    }

    @Override
    public LongDataBufferSlice put(DataBuffer<Long> src) {
        return (LongDataBufferSlice)super.put(src);
    }

    @Override
    protected LongDataBuffer delegate() {
        return (LongDataBuffer)super.delegate();
    }

    private LongDataBufferSlice(LongDataBuffer buffer, long start, long end) {
        super(buffer, start, end);
    }
}
