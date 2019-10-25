package org.tensorflow.nio.nd.impl.dense;

import java.util.function.BiConsumer;
import org.tensorflow.nio.buffer.slice.DataBufferSlice;
import org.tensorflow.nio.nd.ElementCursor;
import org.tensorflow.nio.nd.NdArray;
import org.tensorflow.nio.nd.impl.dimension.DimensionalSpace;

class DefaultElementCursor<T, U extends NdArray<T>> implements ElementCursor<U> {

  @Override
  public void forEachIdx(BiConsumer<long[], U> consumer) {
    long[] coords = new long[dimensionIdx + 1];

    DimensionalSpace elementDims = array.dimensions().truncateFrom(coords.length);
    DataBufferSlice<T> elementBufferSlice = array.buffer().mutableSlice();
    //long originalStart = elementBufferSlice.start();
    U element = array.allocate(elementBufferSlice, elementDims);

    while (true) {
      long elementPos = array.dimensions().positionOf(coords, false);
      elementBufferSlice.moveTo(elementPos);//originalStart + elementPos);
      consumer.accept(coords, element);
      int j;
      for (j = dimensionIdx; j >= 0; --j) {
        if ((coords[j] = (coords[j] + 1) % array.shape().size(j)) > 0) {
          break;
        }
      }
      if (j < 0) {
        return;
      }
    }
  }

  DefaultElementCursor(int dimensionIdx, AbstractDenseNdArray<T, U> array) {
    this.dimensionIdx = dimensionIdx;
    this.array = array;
  }

  private final int dimensionIdx;
  private final AbstractDenseNdArray<T, U> array;
}
