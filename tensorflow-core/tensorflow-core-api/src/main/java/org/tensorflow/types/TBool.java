package org.tensorflow.types;

import java.nio.ByteBuffer;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.converter.BooleanDataConverter;
import org.tensorflow.nio.buffer.impl.logical.BooleanLogicalDataBuffer;
import org.tensorflow.nio.nd.BooleanNdArray;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dense.BooleanDenseNdArray;
import org.tensorflow.types.family.TType;

public interface TBool extends BooleanNdArray, TType {

  DataType<TBool> DTYPE = DataType.create("BOOL", 10, 1, TBoolImpl::map);

  static Tensor<TBool> scalar(boolean value) {
    Tensor<TBool> t = tensorOfShape();
    t.data().setBoolean(value);
    return t;
  }

  static Tensor<TBool> vector(boolean... values) {
    Tensor<TBool> t = tensorOfShape(values.length);
    t.data().write(values);
    return t;
  }

  static Tensor<TBool> tensorOfShape(long... dimensionSizes) {
    return Tensor.allocate(DTYPE, Shape.make(dimensionSizes));
  }
}

class TBoolImpl extends BooleanDenseNdArray implements TBool {

  static TBool map(ByteBuffer[] tensorBuffers, Shape shape) {
    BooleanDataBuffer[] buffers = new BooleanDataBuffer[tensorBuffers.length];
    for (int i = 0; i < tensorBuffers.length; ++i) {
      buffers[i] = BooleanLogicalDataBuffer.map(DataBuffers.wrap(tensorBuffers[i]), BOOL_MAPPER);
    }
    return new TBoolImpl(DataBuffers.join(buffers), shape);
  }

  private TBoolImpl(BooleanDataBuffer buffer, Shape shape) {
    super(buffer, shape);
  }

  private static BooleanDataConverter BOOL_MAPPER = new BooleanDataConverter() {

    @Override
    public void writeBoolean(ByteDataBuffer buffer, boolean value) {
      buffer.put((byte)(value ? 1 : 0));
    }

    @Override
    public boolean readBoolean(ByteDataBuffer buffer) {
      return buffer.get() > 0;
    }

    @Override
    public int sizeInBytes() {
      return TBool.DTYPE.byteSize();
    }
  };
}
