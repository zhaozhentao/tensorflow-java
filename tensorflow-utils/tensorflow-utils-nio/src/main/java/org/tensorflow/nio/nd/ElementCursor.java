package org.tensorflow.nio.nd;

import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * Iterates through a sequence of elements of an N-dimensional array.
 *
 * @param <T> data type of the array being iterated
 */
public interface ElementCursor<T extends NdArray<?>> {

  /**
   * Visit each elements of this iteration.
   *
   * <p><i>Important: the consumer method should not keep a reference to the elements of the
   * iteration, as they might reuse the same object with just a different state for better
   * performance.</i>
   *
   * @param consumer method to invoke for each elements
   */
  default void forEach(Consumer<T> consumer) {
    forEachIdx((c, e) -> consumer.accept(e));
  }

  /**
   * Visit each elements of this iteration and their respective coordinates.
   *
   * <p><i>Important: the consumer method should not keep a reference to the elements or the
   * coordinates of the iteration, as they might reuse the same object with just a different state
   * for better performance.</i>
   *
   * @param consumer method to invoke for each elements
   */
  void forEachIdx(BiConsumer<long[], T> consumer);
}
