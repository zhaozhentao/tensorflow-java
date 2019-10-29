package org.tensorflow.types;

import java.nio.ByteBuffer;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.nd.ByteNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dense.ByteDenseNdArray;
import org.tensorflow.types.family.TNumber;

public interface TUInt8 extends ByteNdArray, TNumber {

  DataType<TUInt8> DTYPE = DataType.create("UINT8", 4, 1, TUInt8Impl::map);

  static Tensor<TUInt8> scalar(byte value) {
    Tensor<TUInt8> t = tensorOfShape();
    t.data().setByte(value);
    return t;
  }

  static Tensor<TUInt8> vector(byte... values) {
    Tensor<TUInt8> t = tensorOfShape(values.length);
    t.data().write(values);
    return t;
  }

  static Tensor<TUInt8> tensorOfShape(long... dimensionSizes) {
    return Tensor.allocate(DTYPE, Shape.make(dimensionSizes));
  }
}

class TUInt8Impl extends ByteDenseNdArray implements TUInt8 {

  static TUInt8 map(ByteBuffer[] tensorBuffers, Shape shape) {
    ByteDataBuffer[] buffers = new ByteDataBuffer[tensorBuffers.length];
    for (int i = 0; i < tensorBuffers.length; ++i) {
      buffers[i] = DataBuffers.wrap(tensorBuffers[i]);
    }
    return new TUInt8Impl(DataBuffers.join(buffers), shape);
  }

  private TUInt8Impl(ByteDataBuffer buffer, Shape shape) {
    super(buffer, shape);
  }
}
