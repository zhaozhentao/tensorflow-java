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
package org.tensorflow.nio.nd;

import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.nd.impl.dense.BooleanDenseNdArray;
import org.tensorflow.nio.nd.impl.dense.ByteDenseNdArray;
import org.tensorflow.nio.nd.impl.dense.DenseNdArray;
import org.tensorflow.nio.nd.impl.dense.DoubleDenseNdArray;
import org.tensorflow.nio.nd.impl.dense.FloatDenseNdArray;
import org.tensorflow.nio.nd.impl.dense.IntDenseNdArray;
import org.tensorflow.nio.nd.impl.dense.LongDenseNdArray;

public final class NdArrays {

  // Byte arrays

  public static ByteNdArray scalar(byte value) {
    return ofBytes(Shape.scalar()).setByte(value);
  }

  public static ByteNdArray vector(byte... values) {
    if (values == null) {
      throw new IllegalArgumentException();
    }
    return ofBytes(Shape.make(values.length)).write(values);
  }
  
  public static ByteNdArray ofBytes(Shape shape) {
    return wrap(DataBuffers.ofBytes(shape.size()), shape);
  }

  public static ByteNdArray wrap(ByteDataBuffer buffer, Shape shape) {
    return ByteDenseNdArray.create(buffer, shape);
  }

  // Long arrays

  public static LongNdArray scalar(long value) {
    return ofLongs(Shape.scalar()).setLong(value);
  }

  public static LongNdArray vector(long... values) {
    if (values == null) {
      throw new IllegalArgumentException();
    }
    return ofLongs(Shape.make(values.length)).write(values);
  }

  public static LongNdArray ofLongs(Shape shape) {
    return wrap(DataBuffers.ofLongs(shape.size()), shape);
  }

  public static LongNdArray wrap(LongDataBuffer buffer, Shape shape) {
    return LongDenseNdArray.create(buffer, shape);
  }

  // Int arrays

  public static IntNdArray scalar(int value) {
    return ofInts(Shape.scalar()).setInt(value);
  }

  public static IntNdArray vector(int... values) {
    if (values == null) {
      throw new IllegalArgumentException();
    }
    return ofInts(Shape.make(values.length)).write(values);
  }

  public static IntNdArray ofInts(Shape shape) {
    return wrap(DataBuffers.ofIntegers(shape.size()), shape);
  }

  public static IntNdArray wrap(IntDataBuffer buffer, Shape shape) {
    return IntDenseNdArray.create(buffer, shape);
  }

  // Float arrays

  public static FloatNdArray scalar(float value) {
    return ofFloats(Shape.scalar()).setFloat(value);
  }

  public static FloatNdArray vector(float... values) {
    if (values == null) {
      throw new IllegalArgumentException();
    }
    return ofFloats(Shape.make(values.length)).write(values);
  }

  public static FloatNdArray ofFloats(Shape shape) {
    return wrap(DataBuffers.ofFloats(shape.size()), shape);
  }

  public static FloatNdArray wrap(FloatDataBuffer buffer, Shape shape) {
    return FloatDenseNdArray.create(buffer, shape);
  }

  // Double arrays

  public static DoubleNdArray scalar(double value) {
    return ofDoubles(Shape.scalar()).setDouble(value);
  }

  public static DoubleNdArray vector(double... values) {
    if (values == null) {
      throw new IllegalArgumentException();
    }
    return ofDoubles(Shape.make(values.length)).write(values);
  }

  public static DoubleNdArray ofDoubles(Shape shape) {
    return wrap(DataBuffers.ofDoubles(shape.size()), shape);
  }

  public static DoubleNdArray wrap(DoubleDataBuffer buffer, Shape shape) {
    return DoubleDenseNdArray.create(buffer, shape);
  }

  // Boolean arrays

  public static BooleanNdArray scalar(boolean value) {
    return ofBooleans(Shape.scalar()).setBoolean(value);
  }

  public static BooleanNdArray vector(boolean... values) {
    if (values == null) {
      throw new IllegalArgumentException();
    }
    return ofBooleans(Shape.make(values.length)).write(values);
  }

  public static BooleanNdArray ofBooleans(Shape shape) {
    return wrap(DataBuffers.ofBooleans(shape.size()), shape);
  }

  public static BooleanNdArray wrap(BooleanDataBuffer buffer, Shape shape) {
    return BooleanDenseNdArray.create(buffer, shape);
  }

  // Object arrays

  @SuppressWarnings("unchecked")
  public static <T> NdArray<T> scalarOf(T value) {
    if (value == null) {
      throw new IllegalArgumentException();
    }
    return of((Class<T>)value.getClass(), Shape.scalar()).setValue(value);
  }

  @SuppressWarnings("unchecked")
  public static <T> NdArray<T> vectorOf(T... values) {
    if (values == null) {
      throw new IllegalArgumentException();
    }
    return of((Class<T>)values[0].getClass(), Shape.make(values.length)).write(values);
  }

  public static <T> NdArray<T> of(Class<T> clazz, Shape shape) {
    return wrap(DataBuffers.of(clazz, shape.size()), shape);
  }

  public static <T> NdArray<T> wrap(DataBuffer<T> buffer, Shape shape) {
    return DenseNdArray.wrap(buffer, shape);
  }
}

