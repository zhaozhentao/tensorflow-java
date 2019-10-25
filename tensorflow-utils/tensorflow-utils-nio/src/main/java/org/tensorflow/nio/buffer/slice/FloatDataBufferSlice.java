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

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer;

public class FloatDataBufferSlice extends DataBufferSlice<Float> implements FloatDataBuffer {

    public FloatDataBufferSlice(FloatDataBuffer buffer) {
        super(buffer);
    }

    @Override
    public float getFloat() {
        return delegate().getFloat();
    }

    @Override
    public float getFloat(long index) {
        return delegate().getFloat(offset(index));
    }

    @Override
    public FloatDataBufferSlice get(float[] dst, int offset, int length) {
        delegate().get(dst, offset, length);
        return this;
    }

    @Override
    public FloatDataBufferSlice putFloat(float value) {
        delegate().putFloat(value);
        return this;
    }

    @Override
    public FloatDataBufferSlice putFloat(long index, float value) {
        delegate().putFloat(offset(index), value);
        return this;
    }

    @Override
    public FloatDataBufferSlice put(float[] src, int offset, int length) {
        delegate().put(src, offset, length);
        return this;
    }

    @Override
    public FloatDataBufferSlice duplicate() {
        return new FloatDataBufferSlice(delegate(), start(), end());
    }

    @Override
    public FloatDataBufferSlice limit(long newLimit) {
        return (FloatDataBufferSlice)super.limit(newLimit);
    }

    @Override
    public FloatDataBufferSlice withLimit(long limit) {
        return (FloatDataBufferSlice)super.withLimit(limit);
    }

    @Override
    public FloatDataBufferSlice position(long newPosition) {
        return (FloatDataBufferSlice)super.position(newPosition);
    }

    @Override
    public FloatDataBufferSlice withPosition(long position) {
        return (FloatDataBufferSlice)super.withPosition(position);
    }

    @Override
    public FloatDataBufferSlice rewind() {
        return (FloatDataBufferSlice)super.rewind();
    }

    @Override
    public FloatDataBufferSlice put(Float value) {
        return (FloatDataBufferSlice)super.put(value);
    }

    @Override
    public FloatDataBufferSlice put(long index, Float value) {
        return (FloatDataBufferSlice)super.put(index, value);
    }

    @Override
    public FloatDataBufferSlice put(DataBuffer<Float> src) {
        return (FloatDataBufferSlice)super.put(src);
    }

    @Override
    protected FloatDataBuffer delegate() {
        return (FloatDataBuffer)super.delegate();
    }

    private FloatDataBufferSlice(FloatDataBuffer buffer, long start, long end) {
        super(buffer, start, end);
    }
}
