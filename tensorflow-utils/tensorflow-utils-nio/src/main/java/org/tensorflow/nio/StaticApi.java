package org.tensorflow.nio;

import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.nd.BooleanNdArray;
import org.tensorflow.nio.nd.ByteNdArray;
import org.tensorflow.nio.nd.DoubleNdArray;
import org.tensorflow.nio.nd.FloatNdArray;
import org.tensorflow.nio.nd.IntNdArray;
import org.tensorflow.nio.nd.LongNdArray;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.NdArrays;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Index;
import org.tensorflow.nio.nd.index.Indices;

interface StaticApi {

  static <T> DataBuffer<T> bufferOf(Class<T> clazz, long capacity) {
    return DataBuffers.of(clazz, capacity);
  }

  static ByteDataBuffer bufferOfBytes(long capacity) {
    return DataBuffers.ofBytes(capacity);
  }

  static IntDataBuffer bufferOfInts(long capacity) {
    return DataBuffers.ofIntegers(capacity);
  }

  static LongDataBuffer bufferOfLongs(long capacity) {
    return DataBuffers.ofLongs(capacity);
  }

  static FloatDataBuffer bufferOfFloats(long capacity) {
    return DataBuffers.ofFloats(capacity);
  }

  static DoubleDataBuffer bufferOfDoubles(long capacity) {
    return DataBuffers.ofDoubles(capacity);
  }

  static BooleanDataBuffer bufferOfBooleans(long capacity) {
    return DataBuffers.ofBooleans(capacity);
  }

  static <T> DataBuffer<T> bufferOf(T[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  static ByteDataBuffer bufferOf(byte[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  static IntDataBuffer bufferOf(int[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  static LongDataBuffer bufferOf(long[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  static FloatDataBuffer bufferOf(float[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  static DoubleDataBuffer bufferOf(double[] array, boolean readOnly) {
    return DataBuffers.wrap(array, readOnly);
  }

  static <T> NdArray<T> ndArrayOf(Class<T> clazz, Shape shape) {
    return NdArrays.of(clazz, shape);
  }

  static <T> NdArray<T> ndArrayOf(DataBuffer<T> buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  } 

  static ByteNdArray ndArrayOfBytes(Shape shape) {
    return NdArrays.ofBytes(shape);
  }

  static IntNdArray ndArrayOfInts(Shape shape) {
    return NdArrays.ofInts(shape);
  }

  static LongNdArray ndArrayOfLongs(Shape shape) {
    return NdArrays.ofLongs(shape);
  }

  static FloatNdArray ndArrayOfFloats(Shape shape) {
    return NdArrays.ofFloats(shape);
  }

  static DoubleNdArray ndArrayOfDoubles(Shape shape) {
    return NdArrays.ofDoubles(shape);
  }

  static BooleanNdArray ndArrayOfBooleans(Shape shape) {
    return NdArrays.ofBooleans(shape);
  }

  static ByteNdArray ndArrayOf(ByteDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  static IntNdArray ndArrayOf(IntDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  static LongNdArray ndArrayOf(LongDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  static FloatNdArray ndArrayOf(FloatDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  static DoubleNdArray ndArrayOf(DoubleDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  static BooleanNdArray ndArrayOf(BooleanDataBuffer buffer, Shape shape) {
    return NdArrays.wrap(buffer, shape);
  }

  static ByteNdArray scalar(byte value) {
    return NdArrays.scalar(value);
  }

  static IntNdArray scalar(int value) {
    return NdArrays.scalar(value);
  }

  static LongNdArray scalar(long value) {
    return NdArrays.scalar(value);
  }

  static FloatNdArray scalar(float value) {
    return NdArrays.scalar(value);
  }

  static DoubleNdArray scalar(double value) {
    return NdArrays.scalar(value);
  }

  static BooleanNdArray scalar(boolean value) {
    return NdArrays.scalar(value);
  }

  static ByteNdArray vector(byte... values) {
    return NdArrays.vector(values);
  }

  static IntNdArray vector(int... values) {
    return NdArrays.vector(values);
  }

  static LongNdArray vector(long... values) {
    return NdArrays.vector(values);
  }

  static FloatNdArray vector(float... values) {
    return NdArrays.vector(values);
  }

  static DoubleNdArray vector(double... values) {
    return NdArrays.vector(values);
  }

  static BooleanNdArray vector(boolean... values) {
    return NdArrays.vector(values);
  }

  static Shape shape(long... dimensionSize) {
    return Shape.make(dimensionSize);
  }

  static Index at(long index) {
    return Indices.at(index);
  }

  static Index at(NdArray<? extends Number> index) {
    return Indices.at(index);
  }
  
  static Index all() {
    return Indices.all();
  }
  
  static Index seq(long... indices) {
    return Indices.seq(indices);
  }
  
  static Index elem(NdArray<? extends Number> indices) {
    return Indices.elem(indices);
  }
  
  static Index even() {
    return Indices.even();
  }

  static Index odd() {
    return Indices.odd();
  }
  
  static Index step(long stepLength) {
    return Indices.step(stepLength);
  }
  
  static Index from(long start) {
    return Indices.from(start);
  }

  static Index to(long end) {
    return Indices.to(end);
  }
  
  static Index range(long start, long end) {
    return Indices.range(start, end);
  }

  static Index flip() {
    return Indices.flip();
  }
}
