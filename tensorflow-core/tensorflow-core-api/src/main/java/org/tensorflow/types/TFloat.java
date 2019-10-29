package org.tensorflow.types;

import java.nio.ByteBuffer;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.FloatDataBuffer;
import org.tensorflow.nio.nd.FloatNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dense.FloatDenseNdArray;
import org.tensorflow.types.family.TDecimal;

public interface TFloat extends FloatNdArray, TDecimal {

  DataType<TFloat> DTYPE = DataType.create("FLOAT", 1, 4, TFloatImpl::map);

  static Tensor<TFloat> scalar(float value) {
    Tensor<TFloat> t = tensorOfShape();
    t.data().setFloat(value);
    return t;
  }

  static Tensor<TFloat> vector(float... values) {
    Tensor<TFloat> t = tensorOfShape(values.length);
    t.data().write(values);
    return t;
  }

  static Tensor<TFloat> tensorOfShape(long... dimensionSizes) {
    return Tensor.allocate(DTYPE, Shape.make(dimensionSizes));
  }
}

class TFloatImpl extends FloatDenseNdArray implements TFloat {

  static TFloat map(ByteBuffer[] tensorBuffers, Shape shape) {
    FloatDataBuffer[] buffers = new FloatDataBuffer[tensorBuffers.length];
    for (int i = 0; i < tensorBuffers.length; ++i) {
      buffers[i] = DataBuffers.wrap(tensorBuffers[i].asFloatBuffer());
    }
    return new TFloatImpl(DataBuffers.join(buffers), shape);
  }

  private TFloatImpl(FloatDataBuffer buffer, Shape shape) {
    super(buffer, shape);
  }
}

