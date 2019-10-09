package org.tensorflow.types;

import java.nio.ByteBuffer;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.nio.buffer.IntDataBuffer;
import org.tensorflow.nio.buffer.impl.single.IntJdkDataBuffer;
import org.tensorflow.nio.nd.IntNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dense.IntDenseNdArray;
import org.tensorflow.types.family.TNumber;

public interface TInt32 extends IntNdArray, TNumber {

  DataType<TInt32> DTYPE = DataType.create("INT32", 3, 4, TInt32Impl::map);

  static Tensor<TInt32> scalar(int value) {
    Tensor<TInt32> t = tensorOfShape(Shape.scalar());
    t.data().set(value);
    return t;
  }

  static Tensor<TInt32> vector(int... values) {
    Tensor<TInt32> t = tensorOfShape(Shape.make(values.length));
    t.data().write(values);
    return t;
  }

  static Tensor<TInt32> tensorOfShape(Shape shape) {
    return Tensor.allocate(DTYPE, shape);
  }
}

class TInt32Impl extends IntDenseNdArray implements TInt32 {

  static TInt32 map(ByteBuffer[] tensorBuffers, Shape shape) {
    IntDataBuffer buffer = BufferUtils.toIntDataBuffer(tensorBuffers, b ->
        IntJdkDataBuffer.wrap(b.asIntBuffer())
    );
    return new TInt32Impl(buffer, shape);
  }

  private TInt32Impl(IntDataBuffer buffer, Shape shape) {
    super(buffer, shape);
  }
}

