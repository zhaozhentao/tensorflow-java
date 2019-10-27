package org.tensorflow.types;

import com.google.common.base.Charsets;
import java.nio.ByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Stream;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;
import org.tensorflow.nio.buffer.ByteDataBuffer;
import org.tensorflow.nio.buffer.DataBuffer;
import org.tensorflow.nio.buffer.DataBuffers;
import org.tensorflow.nio.buffer.LongDataBuffer;
import org.tensorflow.nio.buffer.impl.AbstractBasicDataBuffer;
import org.tensorflow.nio.buffer.impl.Validator;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.NdArrays;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.impl.dense.DenseNdArray;
import org.tensorflow.types.family.TType;

public interface TString extends NdArray<String>, TType {

  DataType<TString> DTYPE = DataType.create("STRING", 7, -1, TStringImpl::map);

  static Tensor<TString> scalar(String value) {
    return copyOf(NdArrays.of(String.class, Shape.scalar()).setValue(value));
  }

  static Tensor<TString> vector(String... values) {
    return copyOf(NdArrays.of(String.class, Shape.make(values.length)).write(values));
  }

  static Tensor<TString> copyOf(NdArray<String> src) {
    return TStringImpl.createTensor(src);
  }
}

class TStringImpl extends DenseNdArray<String> implements TString {

  static Tensor<TString> createTensor(NdArray<String> src) {

    // First, compute the capacity of the tensor to create
    AtomicLong capacity = new AtomicLong(src.size() * 8);  // add space to store 64-bits offsets
    src.scalars().forEach(e -> {
      byte[] bytes = e.getValue().getBytes(Charsets.UTF_8);
      capacity.addAndGet(bytes.length + varintLength(bytes.length));  // add space to store value + length
    });

    // Allocate the tensor of the right capacity and init its data from source array
    Tensor<TString> tensor = Tensor.allocate(TString.DTYPE, src.shape(), capacity.get());
    ((TStringImpl)tensor.data()).buffer().init(src);
    return tensor;
  }

  static TString map(ByteBuffer[] tensorBuffers, Shape shape) {
    if (shape.size() > tensorBuffers[0].capacity()) {
      throw new IllegalArgumentException(); // FIXME!
    }
    LongDataBuffer offsetBuffer = DataBuffers.wrap(tensorBuffers[0].asLongBuffer())
        .limit(shape.size())
        .slice();
    ByteDataBuffer[] dataBuffers = new ByteDataBuffer[tensorBuffers.length];
    dataBuffers[0] = DataBuffers.wrap(tensorBuffers[0])
        .position(offsetBuffer.capacity() * Long.BYTES)
        .slice();
    for (int i = 1; i < tensorBuffers.length; ++i) {
      dataBuffers[i] = DataBuffers.wrap(tensorBuffers[i]);
    }
    return new TStringImpl(new TStringBuffer(offsetBuffer, DataBuffers.join(dataBuffers)), shape);
  }

  @Override
  @SuppressWarnings("unchecked")
  protected TStringBuffer buffer() {
    return (TStringBuffer)super.buffer();
  }

  private TStringImpl(DataBuffer<String> buffer, Shape shape) {
    super(buffer, shape);
  }

  private static int varintLength(int length) {
    int len = 1;
    while (length >= 0x80) {
      length >>= 7;
      len++;
    }
    return len;
  }
}

class TStringBuffer extends AbstractBasicDataBuffer<String, DataBuffer<String>> implements DataBuffer<String> {

  @Override
  public long capacity() {
    return capacity;
  }

  @Override
  public String get() {
    return get(nextPosition());
  }

  @Override
  public String get(long index) {
    Validator.getArgs(this, index);
    long offset = offsets.get(index);
    data.position(offset);
    int length = decodeVarint(data);
    byte[] bytes = new byte[length];
    data.get(bytes);
    return new String(bytes, Charsets.UTF_8);
  }

  @Override
  public Stream<String> stream() {
    throw new UnsupportedOperationException("Tensors of strings does not support data streaming");
  }

  @Override
  public DataBuffer<String> put(String value) {
    throw new ReadOnlyBufferException();
  }

  @Override
  public DataBuffer<String> put(long index, String value) {
    throw new ReadOnlyBufferException();
  }

  @Override
  public DataBuffer<String> duplicate() {
    return new TStringBuffer(offsets, data, position(), limit());
  }

  void init(NdArray<String> src) {
    src.scalars().forEach(e -> {
      String value = e.getValue();
      long offset = data.position();
      offsets.put(offset);
      encodeVarint(data, value.length());
      data.put(value.getBytes(Charsets.UTF_8));
    });
    offsets.rewind();
    data.rewind();
  }

  private final LongDataBuffer offsets;
  private final ByteDataBuffer data;
  private final long capacity;

  TStringBuffer(LongDataBuffer offsets, ByteDataBuffer data) {
    this(offsets, data, 0, data.capacity());
  }

  private TStringBuffer(LongDataBuffer offsets, ByteDataBuffer data, long position, long limit) {
    super(true, position, limit);
    this.offsets = offsets;
    this.data = data;
    this.capacity = offsets.capacity();
  }

  private static void encodeVarint(ByteDataBuffer buffer, int value) {
    int v = value;
    while (v >= 0x80) {
      buffer.put((byte)((v & 0x7F) | 0x80));
      v >>= 7;
    }
    buffer.put((byte)v);
  }

  private static int decodeVarint(ByteDataBuffer buffer) {
    byte b;
    int pos = 0;
    int v = 0;
    do {
      b = buffer.get();
      v |= (b & 0x7F) << pos++;
    } while ((b & 0x80) != 0);
    return v;
  }
}
