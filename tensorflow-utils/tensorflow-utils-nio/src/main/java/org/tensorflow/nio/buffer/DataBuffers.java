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
import org.tensorflow.nio.buffer.converter.BooleanDataConverter;
import org.tensorflow.nio.buffer.converter.DataConverter;
import org.tensorflow.nio.buffer.converter.DoubleDataConverter;
import org.tensorflow.nio.buffer.converter.FloatDataConverter;
import org.tensorflow.nio.buffer.converter.IntDataConverter;
import org.tensorflow.nio.buffer.converter.LongDataConverter;
import org.tensorflow.nio.buffer.impl.large.BooleanLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.ByteLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.DoubleLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.FloatLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.IntLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.LargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.LongLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.BooleanLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.DoubleLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.FloatLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.IntLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.LogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.LongLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.single.ArrayDataBuffer;
import org.tensorflow.nio.buffer.impl.single.BitSetDataBuffer;
import org.tensorflow.nio.buffer.impl.single.BooleanArrayDataBuffer;
import org.tensorflow.nio.buffer.impl.single.ByteJdkDataBuffer;
import org.tensorflow.nio.buffer.impl.single.DoubleJdkDataBuffer;
import org.tensorflow.nio.buffer.impl.single.FloatJdkDataBuffer;
import org.tensorflow.nio.buffer.impl.single.IntJdkDataBuffer;
import org.tensorflow.nio.buffer.impl.single.LongJdkDataBuffer;

/**
 * Helper class for creating {@code DataBuffer} instances.
 */
public final class DataBuffers {

  /**
   * Creates a buffer of bytes that can store up to {@code capacity} values
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

  /**
   * Join multiple byte buffers together to create a large buffer indexable with 64-bits values.
   *
   * @param buffers buffers to join
   * @return a potentially large buffer
   */
  public static ByteDataBuffer join(ByteDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : ByteLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of longs that can store up to {@code capacity} values
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

  /**
   * Creates a logical buffer of longs that can store up to {@code capacity} values.
   *
   * <p>The provided converter is used to map the long values to/from bytes, allowing custom
   * representation of a long.
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static LongDataBuffer ofLongs(long capacity, LongDataConverter converter) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * converter.sizeInBytes());
    return LongLogicalDataBuffer.map(physicalBuffer, converter);
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

  /**
   * Join multiple long buffers together to create a large buffer indexable with 64-bits values.
   *
   * @param buffers buffers to join
   * @return a potentially large buffer
   */
  public static LongDataBuffer join(LongDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : LongLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of integers that can store up to {@code capacity} values
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static IntDataBuffer ofInts(long capacity) {
    if (capacity > IntJdkDataBuffer.MAX_CAPACITY) {
      return IntLargeDataBuffer.allocate(capacity);
    }
    return IntJdkDataBuffer.allocate(capacity);
  }

  /**
   * Creates a logical buffer of integers that can store up to {@code capacity} values.
   *
   * <p>The provided converter is used to map the integer values to/from bytes, allowing custom
   * representation of a integer.
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static IntDataBuffer ofInts(long capacity, IntDataConverter converter) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * converter.sizeInBytes());
    return IntLogicalDataBuffer.map(physicalBuffer, converter);
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

  /**
   * Join multiple integer buffers together to create a large buffer indexable with 64-bits values.
   *
   * @param buffers buffers to join
   * @return a potentially large buffer
   */
  public static IntDataBuffer join(IntDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : IntLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of doubles that can store up to {@code capacity} values
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

  /**
   * Creates a logical buffer of doubles that can store up to {@code capacity} values.
   *
   * <p>The provided converter is used to map the double values to/from bytes, allowing custom
   * representation of a double.
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static DoubleDataBuffer ofDoubles(long capacity, DoubleDataConverter converter) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * converter.sizeInBytes());
    return DoubleLogicalDataBuffer.map(physicalBuffer, converter);
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

  /**
   * Join multiple double buffers together to create a large buffer indexable with 64-bits values.
   *
   * @param buffers buffers to join
   * @return a potentially large buffer
   */

  public static DoubleDataBuffer join(DoubleDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : DoubleLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of floats that can store up to {@code capacity} values
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

  /**
   * Creates a logical buffer of floats that can store up to {@code capacity} values.
   *
   * <p>The provided converter is used to map the float values to/from bytes, allowing custom
   * representation of a float.
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static FloatDataBuffer ofFloats(long capacity, FloatDataConverter converter) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * converter.sizeInBytes());
    return FloatLogicalDataBuffer.map(physicalBuffer, converter);
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

  /**
   * Join multiple float buffers together to create a large buffer indexable with 64-bits values.
   *
   * @param buffers buffers to join
   * @return a potentially large buffer
   */
  public static FloatDataBuffer join(FloatDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : FloatLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of booleans that can store up to {@code capacity} values
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

  /**
   * Creates a logical buffer of booleans that can store up to {@code capacity} values.
   *
   * <p>The provided converter is used to map the boolean values to/from bytes, allowing custom
   * representation of a boolean.
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static BooleanDataBuffer ofBooleans(long capacity, BooleanDataConverter converter) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * converter.sizeInBytes());
    return BooleanLogicalDataBuffer.map(physicalBuffer, converter);
  }

  /**
   * Wraps an array of booleans into a data buffer.
   *
   * @param array array to wrap
   * @param readOnly true if the buffer created must be read-only
   * @return a new buffer
   */
  public static BooleanDataBuffer wrap(boolean[] array, boolean readOnly) {
    return BooleanArrayDataBuffer.wrap(array, readOnly);
  }

  /**
   * Join multiple boolean buffers together to create a large buffer indexable with 64-bits values.
   *
   * @param buffers buffers to join
   * @return a potentially large buffer
   */
  public static BooleanDataBuffer join(BooleanDataBuffer... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : BooleanLargeDataBuffer.join(buffers);
  }

  /**
   * Creates a buffer of objects of type {@code clazz` that can store up to `capacity} values
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

  /**
   * Creates a logical buffer that can store up to {@code capacity} values.
   *
   * <p>The provided converter is used to map the values to/from bytes, allowing custom
   * representation of this buffer type.
   *
   * @param capacity capacity of the buffer to allocate
   * @return a new buffer
   */
  public static <T> DataBuffer<T> of(long capacity, DataConverter<T> converter) {
    ByteDataBuffer physicalBuffer = ofBytes(capacity * converter.sizeInBytes());
    return LogicalDataBuffer.map(physicalBuffer, converter);
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

  /**
   * Join multiple buffers together to create a large buffer indexable with 64-bits values.
   *
   * @param buffers buffers to join
   * @return a potentially large buffer
   */
  @SafeVarargs
  public static <T> DataBuffer<T> join(DataBuffer<T>... buffers) {
    if (buffers == null) {
      return null;
    }
    return (buffers.length == 1) ? buffers[0] : LargeDataBuffer.join(buffers);
  }
}
