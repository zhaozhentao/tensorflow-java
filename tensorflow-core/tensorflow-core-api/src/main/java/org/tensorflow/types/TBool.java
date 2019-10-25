package org.tensorflow.types;

import java.nio.ByteBuffer;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.nio.buffer.BooleanDataBuffer;
import org.tensorflow.nio.buffer.converter.BooleanDataConverter;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.impl.logical.BooleanLogicalDataBuffer;
import org.tensorflow.nio.buffer.impl.single.ByteJdkDataBuffer;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.BooleanNdArrayImpl;
import org.tensorflow.types.family.TType;

public interface TBool extends org.tensorflow.nio.nd.BooleanNdArray, TType {

  DataType<TBool> DTYPE = DataType.create("BOOL", 10, 1, TBoolImpl::map);

  static Tensor<TBool> scalar(boolean value) {
    Tensor<TBool> t = tensorOfShape(Shape.scalar());
    t.data().setBoolean(value);
    return t;
  }

  static Tensor<TBool> vector(boolean... values) {
    Tensor<TBool> t = tensorOfShape(Shape.make(values.length));
    t.data().write(values);
    return t;
  }

  static Tensor<TBool> tensorOfShape(Shape shape) {
    return Tensor.allocate(DTYPE, shape);
  }
}

class TBoolImpl extends BooleanNdArrayImpl implements TBool {

  static TBool map(ByteBuffer[] tensorBuffers, Shape shape) {
    BooleanDataBuffer buffer = BufferUtils.toBooleanDataBuffer(tensorBuffers, b ->
        BooleanLogicalDataBuffer.map(ByteJdkDataBuffer.wrap(b), BOOL_MAPPER)
    );
    return new TBoolImpl(buffer, shape);
  }

  private TBoolImpl(BooleanDataBuffer buffer, Shape shape) {
    super(buffer, shape);
  }

  private static BooleanDataConverter BOOL_MAPPER = new BooleanDataConverter() {

    @Override
    public void writeBoolean(ByteDataBuffer physicalBuffer, boolean value) {
      physicalBuffer.put((byte)(value ? 1 : 0));
    }

    @Override
    public boolean readBoolean(ByteDataBuffer physicalBuffer) {
      return physicalBuffer.get() > 0;
    }

    @Override
    public int sizeInBytes() {
      return TBool.DTYPE.byteSize();
    }
  };
}
