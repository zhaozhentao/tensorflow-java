package org.tensorflow.types;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.function.Function;
import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.buffer.impl.large.BooleanLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.ByteLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.DoubleLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.FloatLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.IntLargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.LargeDataBuffer;
import org.tensorflow.nio.buffer.impl.large.LongLargeDataBuffer;

final class BufferUtils {

  static ByteDataBuffer toByteDataBuffer(ByteBuffer[] bufs, Function<ByteBuffer, ByteDataBuffer> bufferMapper) {
    ByteDataBuffer[] buffers = Arrays.stream(bufs).map(bufferMapper).toArray(ByteDataBuffer[]::new);
    return (buffers.length == 1) ? buffers[0] : ByteLargeDataBuffer.join(buffers);
  }

  static IntDataBuffer toIntDataBuffer(ByteBuffer[] bufs, Function<ByteBuffer, IntDataBuffer> bufferMapper) {
    IntDataBuffer[] buffers = Arrays.stream(bufs).map(bufferMapper).toArray(IntDataBuffer[]::new);
    return (buffers.length == 1) ? buffers[0] : IntLargeDataBuffer.join(buffers);
  }

  static LongDataBuffer toLongDataBuffer(ByteBuffer[] bufs, Function<ByteBuffer, LongDataBuffer> bufferMapper) {
    LongDataBuffer[] buffers = Arrays.stream(bufs).map(bufferMapper).toArray(LongDataBuffer[]::new);
    return (buffers.length == 1) ? buffers[0] : LongLargeDataBuffer.join(buffers);
  }

  static FloatDataBuffer toFloatDataBuffer(ByteBuffer[] bufs, Function<ByteBuffer, FloatDataBuffer> bufferMapper) {
    FloatDataBuffer[] buffers = Arrays.stream(bufs).map(bufferMapper).toArray(FloatDataBuffer[]::new);
    return (buffers.length == 1) ? buffers[0] : FloatLargeDataBuffer.join(buffers);
  }

  static DoubleDataBuffer toDoubleDataBuffer(ByteBuffer[] bufs, Function<ByteBuffer, DoubleDataBuffer> bufferMapper) {
    DoubleDataBuffer[] buffers = Arrays.stream(bufs).map(bufferMapper).toArray(DoubleDataBuffer[]::new);
    return (buffers.length == 1) ? buffers[0] : DoubleLargeDataBuffer.join(buffers);
  }

  static BooleanDataBuffer toBooleanDataBuffer(ByteBuffer[] bufs, Function<ByteBuffer, BooleanDataBuffer> bufferMapper) {
    BooleanDataBuffer[] buffers = Arrays.stream(bufs).map(bufferMapper).toArray(BooleanDataBuffer[]::new);
    return (buffers.length == 1) ? buffers[0] : BooleanLargeDataBuffer.join(buffers);
  }

  static <T> DataBuffer<T> toDataBuffer(ByteBuffer[] bufs, Function<ByteBuffer, DataBuffer<T>> bufferMapper) {
    DataBuffer<T>[] buffers = Arrays.stream(bufs).map(bufferMapper).toArray(DataBuffer[]::new);
    return (buffers.length == 1) ? buffers[0] : LargeDataBuffer.join(buffers);
  }

  private BufferUtils() {
  }
}
