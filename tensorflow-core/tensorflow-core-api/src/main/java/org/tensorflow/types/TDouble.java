package org.tensorflow.types;

import java.nio.ByteBuffer;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.DoubleDataBuffer;
import org.tensorflow.nio.nd.DoubleNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dense.DoubleDenseNdArray;
import org.tensorflow.types.family.TDecimal;

public interface TDouble extends DoubleNdArray, TDecimal {

  DataType<TDouble> DTYPE = DataType.create("DOUBLE", 2, 8, TDoubleImpl::map);

  static Tensor<TDouble> scalar(double value) {
    Tensor<TDouble> t = tensorOfShape();
    t.data().setDouble(value);
    return t;
  }

  static Tensor<TDouble> vector(double... values) {
    Tensor<TDouble> t = tensorOfShape(values.length);
    t.data().write(values);
    return t;
  }

  static Tensor<TDouble> tensorOfShape(long... dimensionSizes) {
    return Tensor.allocate(DTYPE, Shape.make(dimensionSizes));
  }
}

class TDoubleImpl extends DoubleDenseNdArray implements TDouble {

  static TDouble map(ByteBuffer[] tensorBuffers, Shape shape) {
    DoubleDataBuffer[] buffers = new DoubleDataBuffer[tensorBuffers.length];
    for (int i = 0; i < tensorBuffers.length; ++i) {
      buffers[i] = DataBuffers.wrap(tensorBuffers[i].asDoubleBuffer());
    }
    return new TDoubleImpl(DataBuffers.join(buffers), shape);
  }

  private TDoubleImpl(DoubleDataBuffer buffer, Shape shape) {
    super(buffer, shape);
  }
}
