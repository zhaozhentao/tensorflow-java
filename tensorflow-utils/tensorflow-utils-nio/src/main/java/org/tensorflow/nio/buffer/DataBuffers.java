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
package org.tensorflow.nio.buffer;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

import java.util.Arrays;
import org.tensorflow.nio.buffer.BooleanDataBuffer.BooleanMapper;
import org.tensorflow.nio.buffer.DataBuffer.ValueMapper;
import org.tensorflow.nio.buffer.DoubleDataBuffer.DoubleMapper;
import org.tensorflow.nio.buffer.FloatDataBuffer.FloatMapper;
import org.tensorflow.nio.buffer.IntDataBuffer.IntMapper;
import org.tensorflow.nio.buffer.LongDataBuffer.LongMapper;
import org.tensorflow.nio.buffer.impl.large.BooleanLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.BooleanLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.DoubleLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.FloatLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.IntLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.LogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.LongLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.single.ArrayDataBuffer;
import org.tensorflow.nio.buffer.impl.single.BitSetDataBuffer;
import org.tensorflow.nio.buffer.impl.single.ByteJdkDataBuffer;
import org.tensorflow.nio.buffer.impl.single.DoubleJdkDataBuffer;
import org.tensorflow.nio.buffer.impl.single.FloatJdkDataBuffer;
import org.tensorflow.nio.buffer.impl.single.IntJdkDataBuffer;
import org.tensorflow.nio.buffer.impl.single.LongJdkDataBuffer;
import org.tensorflow.nio.buffer.impl.large.ByteLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.DoubleLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.FloatLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.IntLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.LargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.LongLargeDataBuffer;

/**
 * Helper class for creating `DataBuffer` instances.
 */
public final class DataBuffers {

  /**
   * Creates a buffer of bytes that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static ByteDataBuffer ofBytes(long capacity) {
    if (capacity > ByteJdkDataBuffer.MAX_CAPACITY) {
      return ByteLargeDataBuffer.allocate(capacity);
    }
    return ByteJdkDataBuffer.allocate(capacity);
  }

  /**
   * Wraps an array of bytes into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static ByteDataBuffer wrap(byte[] array, boolean readOnly) {
    ByteBuffer buf = ByteBuffer.wrap(array);
    return ByteJdkDataBuffer.wrap(readOnly ? buf.asReadOnlyBuffer() : buf);
  }

  /**
   * Wraps a JDK byte buffers into a data buffer.
   *
   * @param buf buffer to wrap
   * @return a new buffer
   */
  public static ByteDataBuffer wrap(ByteBuffer buf) {
    return ByteJdkDataBuffer.wrap(buf);
  }

  public static ByteDataBuffer join(ByteDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : ByteLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of longs that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static LongDataBuffer ofLongs(long capacity) {
    if (capacity > LongJdkDataBuffer.MAX_CAPACITY) {
      return LongLargeDataBuffer.allocate(capacity);
    }
    return LongJdkDataBuffer.allocate(capacity);
  }

  public static LongDataBuffer ofLongs(long capacity, LongMapper mapper) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * mapper.sizeInBytes());
    return LongLogicalDataBuffer.map(physicalBuffer, mapper);
  }

  /**
   * Wraps an array of longs into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static LongDataBuffer wrap(long[] array, boolean readOnly) {
    LongBuffer buf = LongBuffer.wrap(array);
    return LongJdkDataBuffer.wrap(readOnly ? buf.asReadOnlyBuffer() : buf);
  }

  /**
   * Wraps a JDK long buffer into a data buffer.
   *
   * @param buf buffer to wrap
   * @return a new buffer
   */
  public static LongDataBuffer wrap(LongBuffer buf) {
    return LongJdkDataBuffer.wrap(buf);
  }

  public static LongDataBuffer join(LongDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : LongLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of integers that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static IntDataBuffer ofIntegers(long capacity) {
    if (capacity > IntJdkDataBuffer.MAX_CAPACITY) {
      return IntLargeDataBuffer.allocate(capacity);
    }
    return IntJdkDataBuffer.allocate(capacity);
  }

  public static IntDataBuffer ofIntegers(long capacity, IntMapper mapper) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * mapper.sizeInBytes());
    return IntLogicalDataBuffer.map(physicalBuffer, mapper);
  }

  /**
   * Wraps an array of integers into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static IntDataBuffer wrap(int[] array, boolean readOnly) {
    IntBuffer buf = IntBuffer.wrap(array);
    return IntJdkDataBuffer.wrap(readOnly ? buf.asReadOnlyBuffer() : buf);
  }

  /**
   * Wraps a JDK integer buffer into a data buffer.
   *
   * @param buf buffer to wrap
   * @return a new buffer
   */
  public static IntDataBuffer wrap(IntBuffer buf) {
    return IntJdkDataBuffer.wrap(buf);
  }

  public static IntDataBuffer join(IntDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : IntLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of doubles that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static DoubleDataBuffer ofDoubles(long capacity) {
    if (capacity > DoubleJdkDataBuffer.MAX_CAPACITY) {
      return DoubleLargeDataBuffer.allocate(capacity);
    }
    return DoubleJdkDataBuffer.allocate(capacity);
  }

  public static DoubleDataBuffer ofDoubles(long capacity, DoubleMapper mapper) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * mapper.sizeInBytes());
    return DoubleLogicalDataBuffer.map(physicalBuffer, mapper);
  }

  /**
   * Wraps an array of doubles into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static DoubleDataBuffer wrap(double[] array, boolean readOnly) {
    DoubleBuffer buf = DoubleBuffer.wrap(array);
    return DoubleJdkDataBuffer.wrap(readOnly ? buf.asReadOnlyBuffer() : buf);
  }

  /**
   * Wraps a JDK double buffer into a data buffer.
   *
   * @param buf buffer to wrap
   * @return a new buffer
   */
  public static DoubleDataBuffer wrap(DoubleBuffer buf) {
    return DoubleJdkDataBuffer.wrap(buf);
  }

  public static DoubleDataBuffer join(DoubleDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : DoubleLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of floats that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static FloatDataBuffer ofFloats(long capacity) {
    if (capacity > FloatJdkDataBuffer.MAX_CAPACITY) {
      return FloatLargeDataBuffer.allocate(capacity);
    }
    return FloatJdkDataBuffer.allocate(capacity);
  }

  public static FloatDataBuffer ofFloats(long capacity, FloatMapper mapper) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * mapper.sizeInBytes());
    return FloatLogicalDataBuffer.map(physicalBuffer, mapper);
  }

  /**
   * Wraps an array of floats into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static FloatDataBuffer wrap(float[] array, boolean readOnly) {
    FloatBuffer buf = FloatBuffer.wrap(array);
    return FloatJdkDataBuffer.wrap(readOnly ? buf.asReadOnlyBuffer() : buf);
  }

  /**
   * Wraps a JDK float buffer into a data buffer.
   *
   * @param buf buffer to wrap
   * @return a new buffer
   */
  public static FloatDataBuffer wrap(FloatBuffer buf) {
    return FloatJdkDataBuffer.wrap(buf);
  }

  public static FloatDataBuffer join(FloatDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : FloatLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of booleans that can store up to `capacity` values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static BooleanDataBuffer ofBooleans(long capacity) {
    if (capacity > BitSetDataBuffer.MAX_CAPACITY) {
      return BooleanLargeDataBuffer.allocate(capacity);
    }
    return BitSetDataBuffer.allocate(capacity);
  }

  public static BooleanDataBuffer ofBooleans(long capacity, BooleanMapper mapper) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * mapper.sizeInBytes());
    return BooleanLogicalDataBuffer.map(physicalBuffer, mapper);
  }

  public static BooleanDataBuffer join(BooleanDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : BooleanLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of objects of type `clazz` that can store up to `capacity` values
   *
   * @param clazz the type of object stored in this buffer
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static <T> DataBuffer<T> of(Class<T> clazz, long capacity) {
    if (capacity > ArrayDataBuffer.MAX_CAPACITY) {
      return LargeDataBuffer.allocate(clazz, capacity);
    }
    return ArrayDataBuffer.allocate(clazz, capacity);
  }

  public static <T> DataBuffer<T> of(long capacity, ValueMapper<T> mapper) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * mapper.sizeInBytes());
    return LogicalDataBuffer.map(physicalBuffer, mapper);
  }
  /**
   * Wraps an array of objects into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static <T> DataBuffer<T> wrap(T[] array, boolean readOnly) {
    return ArrayDataBuffer.wrap(array, readOnly);
  }

  public static <T> DataBuffer<T> join(DataBuffer<T>... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : LargeDataBuffer.join(buffers);
  }
}
