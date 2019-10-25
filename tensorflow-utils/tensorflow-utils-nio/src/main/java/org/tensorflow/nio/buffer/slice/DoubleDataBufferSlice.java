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

import java.util.stream.DoubleStream;

import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;

public class DoubleDataBufferSlice extends DataBufferSlice<Double> implements DoubleDataBuffer {

    public DoubleDataBufferSlice(DoubleDataBuffer buffer) {
        super(buffer);
    }

    @Override
    public DoubleStream doubleStream() {
        // TODO
        throw new UnsupportedOperationException();
    }

    @Override
    public double getDouble() {
        return delegate().getDouble();
    }

    @Override
    public double getDouble(long index) {
        return delegate().getDouble(offset(index));
    }

    @Override
    public DoubleDataBufferSlice get(double[] dst, int offset, int length) {
        delegate().get(dst, offset, length);
        return this;
    }

    @Override
    public DoubleDataBufferSlice putDouble(double value) {
        delegate().putDouble(value);
        return this;
    }

    @Override
    public DoubleDataBufferSlice putDouble(long index, double value) {
        delegate().putDouble(offset(index), value);
        return this;
    }

    @Override
    public DoubleDataBufferSlice put(double[] src, int offset, int length) {
        delegate().put(src, offset, length);
        return this;
    }

    @Override
    public DoubleDataBufferSlice duplicate() {
        return new DoubleDataBufferSlice(delegate().duplicate(), start(), end());
    }

    @Override
    public DoubleDataBufferSlice limit(long newLimit) {
        return (DoubleDataBufferSlice)super.limit(newLimit);
    }

    @Override
    public DoubleDataBufferSlice withLimit(long limit) {
        return (DoubleDataBufferSlice)super.withLimit(limit);
    }

    @Override
    public DoubleDataBufferSlice position(long newPosition) {
        return (DoubleDataBufferSlice)super.position(newPosition);
    }

    @Override
    public DoubleDataBufferSlice withPosition(long position) {
        return (DoubleDataBufferSlice)super.withPosition(position);
    }

    @Override
    public DoubleDataBufferSlice rewind() {
        return (DoubleDataBufferSlice)super.rewind();
    }

    @Override
    public DoubleDataBufferSlice put(Double value) {
        return (DoubleDataBufferSlice)super.put(value);
    }

    @Override
    public DoubleDataBufferSlice put(long index, Double value) {
        return (DoubleDataBufferSlice)super.put(index, value);
    }

    @Override
    public DoubleDataBufferSlice put(DataBuffer<Double> src) {
        return (DoubleDataBufferSlice)super.put(src);
    }

    @Override
    protected DoubleDataBuffer delegate() {
        return (DoubleDataBuffer)super.delegate();
    }

    private DoubleDataBufferSlice(DoubleDataBuffer buffer, long start, long end) {
        super(buffer, start, end);
    }
}
