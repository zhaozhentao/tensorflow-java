package org.tensorflow.types;

import java.nio.ByteBuffer;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.buffer.impl.single.FloatJdkDataBuffer;
import org.tensorflow.nio.nd.FloatNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dense.FloatDenseNdArray;
import org.tensorflow.types.family.TDecimal;

public interface TFloat extends FloatNdArray, TDecimal {

  DataType<TFloat> DTYPE = DataType.create("FLOAT", 1, 4, TFloatImpl::map);

  static Tensor<TFloat> scalar(float value) {
    Tensor<TFloat> t = tensorOfShape(Shape.scalar());
    t.data().set(value);
    return t;
  }

  static Tensor<TFloat> vector(float... values) {
    Tensor<TFloat> t = tensorOfShape(Shape.make(values.length));
    t.data().write(values);
    return t;
  }

  static Tensor<TFloat> tensorOfShape(Shape shape) {
    return Tensor.allocate(DTYPE, shape);
  }
}

class TFloatImpl extends FloatDenseNdArray implements TFloat {

  static TFloat map(ByteBuffer[] tensorBuffers, Shape shape) {
    FloatDataBuffer buffer = BufferUtils.toFloatDataBuffer(tensorBuffers, b ->
        FloatJdkDataBuffer.wrap(b.asFloatBuffer())
    );
    return new TFloatImpl(buffer, shape);
  }

  private TFloatImpl(FloatDataBuffer buffer, Shape shape) {
    super(buffer, shape);
  }
}

