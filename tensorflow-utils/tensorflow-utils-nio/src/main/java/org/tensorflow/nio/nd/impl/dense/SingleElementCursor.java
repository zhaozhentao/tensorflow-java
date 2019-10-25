package org.tensorflow.nio.nd.impl.dense;

import java.util.function.BiConsumer;
import java.util.function.Consumer;
import org.tensorflow.nio.nd.ElementCursor;
import org.tensorflow.nio.nd.IllegalRankException;
import org.tensorflow.nio.nd.NdArray;

class SingleElementCursor<U extends NdArray<?>> implements ElementCursor<U> {

  @Override
  public void forEach(Consumer<U> consumer) {
    consumer.accept(array);
  }

  @Override
  public void forEachIdx(BiConsumer<long[], U> consumer) {
    throw new IllegalRankException("Single element has no coordinates to iterate on, use forEach()");
  }

  SingleElementCursor(U array) {
    this.array = array;
  }

  private final U array;
}
