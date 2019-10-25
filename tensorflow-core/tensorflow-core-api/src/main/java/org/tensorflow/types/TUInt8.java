package org.tensorflow.types;

import java.nio.ByteBuffer;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.impl.single.ByteJdkDataBuffer;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.ByteNdArray;
import org.tensorflow.types.family.TNumber;

public interface TUInt8 extends org.tensorflow.nio.nd.ByteNdArray, TNumber {

  DataType<TUInt8> DTYPE = DataType.create("UINT8", 4, 1, TUInt8Impl::map);

  static Tensor<TUInt8> scalar(byte value) {
    Tensor<TUInt8> t = tensorOfShape(Shape.scalar());
    t.data().setByte(value);
    return t;
  }

  static Tensor<TUInt8> vector(byte... values) {
    Tensor<TUInt8> t = tensorOfShape(Shape.make(values.length));
    t.data().write(values);
    return t;
  }

  static Tensor<TUInt8> tensorOfShape(Shape shape) {
    return Tensor.allocate(DTYPE, shape);
  }
}

class TUInt8Impl extends ByteNdArray implements TUInt8 {

  static TUInt8 map(ByteBuffer[] tensorBuffers, Shape shape) {
    ByteDataBuffer buffer = BufferUtils.toByteDataBuffer(tensorBuffers, ByteJdkDataBuffer::wrap);
    return new TUInt8Impl(buffer, shape);
  }

  private TUInt8Impl(ByteDataBuffer buffer, Shape shape) {
    super(buffer, shape);
  }
}
