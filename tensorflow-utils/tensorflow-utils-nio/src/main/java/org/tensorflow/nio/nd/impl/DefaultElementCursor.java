package org.tensorflow.nio.nd.impl;

import java.util.function.BiConsumer;
import org.tensorflow.nio.nd.ElementCursor;
import org.tensorflow.nio.nd.NdArray;

class DefaultElementCursor<U extends NdArray<?>> implements ElementCursor<U> {

  @Override
  public void forEachIdx(BiConsumer<long[], U> consumer) {
    long[] coords = new long[dimensionIdx + 1];
    int j;
    do {
      consumer.accept(coords, (U)array.get(coords)); // TODO instead of getting a new array, mutate one
      for (j = dimensionIdx; j >= 0; --j) {
        if ((coords[j] = (coords[j] + 1) % array.shape().size(j)) > 0) {
          break;
        }
      }
    } while (j >= 0);
  }

  DefaultElementCursor(int dimensionIdx, U array) {
    this.dimensionIdx = dimensionIdx;
    this.array = array;
  }

  private final int dimensionIdx;
  private final U array;
}
