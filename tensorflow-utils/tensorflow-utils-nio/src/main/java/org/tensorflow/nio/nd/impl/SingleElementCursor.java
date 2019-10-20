package org.tensorflow.nio.nd.impl;

import java.util.function.BiConsumer;
import java.util.function.Consumer;
import org.tensorflow.nio.nd.ElementCursor;
import org.tensorflow.nio.nd.IllegalRankException;
import org.tensorflow.nio.nd.NdArray;

class SingleElementCursor<T extends NdArray<?>> implements ElementCursor<T> {

  @Override
  public void forEach(Consumer<T> consumer) {
    consumer.accept(array);
  }

  @Override
  public void forEachIdx(BiConsumer<long[], T> consumer) {
    throw new IllegalRankException("Single element has no coordinates to iterate on, use forEach()");
  }

  SingleElementCursor(T array) {
    this.array = array;
  }

  private final T array;
}
