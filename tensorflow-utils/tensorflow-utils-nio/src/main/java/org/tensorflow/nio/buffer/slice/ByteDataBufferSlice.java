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

import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;

public class ByteDataBufferSlice extends DataBufferSlice<Byte> implements ByteDataBuffer {

    public ByteDataBufferSlice(ByteDataBuffer buffer) {
        super(buffer);
    }

    @Override
    public byte getByte() {
        return delegate().getByte();
    }

    @Override
    public byte getByte(long index) {
        return delegate().getByte(offset(index));
    }

    @Override
    public ByteDataBufferSlice get(byte[] dst, int offset, int length) {
        delegate().get(dst, offset, length);
        return this;
    }

    @Override
    public ByteDataBufferSlice putByte(byte value) {
        delegate().putByte(value);
        return this;
    }

    @Override
    public ByteDataBufferSlice putByte(long index, byte value) {
        delegate().putByte(offset(index), value);
        return this;
    }

    @Override
    public ByteDataBufferSlice put(byte[] src, int offset, int length) {
        delegate().put(src, offset, length);
        return this;
    }

    @Override
    public ByteDataBufferSlice duplicate() {
        return new ByteDataBufferSlice(delegate(), start(), end());
    }

    @Override
    public ByteDataBufferSlice limit(long newLimit) {
        return (ByteDataBufferSlice)super.limit(newLimit);
    }

    @Override
    public ByteDataBufferSlice withLimit(long limit) {
        return (ByteDataBufferSlice)super.withLimit(limit);
    }

    @Override
    public ByteDataBufferSlice position(long newPosition) {
        return (ByteDataBufferSlice)super.position(newPosition);
    }

    @Override
    public ByteDataBufferSlice withPosition(long position) {
        return (ByteDataBufferSlice)super.withPosition(position);
    }

    @Override
    public ByteDataBufferSlice rewind() {
        return (ByteDataBufferSlice)super.rewind();
    }

    @Override
    public ByteDataBufferSlice put(Byte value) {
        return (ByteDataBufferSlice)super.put(value);
    }

    @Override
    public ByteDataBufferSlice put(long index, Byte value) {
        return (ByteDataBufferSlice)super.put(index, value);
    }

    @Override
    public ByteDataBufferSlice put(DataBuffer<Byte> src) {
        return (ByteDataBufferSlice)super.put(src);
    }

    @Override
    protected ByteDataBuffer delegate() {
        return (ByteDataBuffer)super.delegate();
    }

    private ByteDataBufferSlice(ByteDataBuffer buffer, long start, long end) {
        super(buffer, start, end);
    }
}
