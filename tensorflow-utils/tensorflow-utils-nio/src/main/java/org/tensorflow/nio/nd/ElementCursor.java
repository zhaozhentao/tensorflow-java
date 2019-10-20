package org.tensorflow.nio.nd;

import java.util.function.BiConsumer;
import java.util.function.Consumer;

public interface ElementCursor<T extends NdArray<?>> {

  default void forEach(Consumer<T> consumer) {
    forEachIdx((c, e) -> consumer.accept(e));
  }

  void forEachIdx(BiConsumer<long[], T> consumer);
}
