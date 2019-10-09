/*
 Copyright 2019 The TensorFlow Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 =======================================================================
 */
package org.tensorflow.nio.nd.impl.dense;

import java.util.function.BiConsumer;
import org.tensorflow.nio.nd.impl.shape.Dimension;

class BulkDataTransfer<R extends AbstractDenseNdArray<?, ?>> {

  static <R extends AbstractDenseNdArray<?, ?>> BulkDataTransfer<R> create(R array) {
    int bulkCopyDimensionIdx = -1;
    long bulkCopySize = 1L;

    // Find what are the biggest chunk of data that we can copy in bulk by starting from the last dimension of this array and
    // iterating backward until we hit a dimension that is segmented (if any)
    for (int i = array.shape().numDimensions() - 1; i >= 0; --i) {
      Dimension dim = array.shape().dimension(i);
      if (dim.isSegmented()) {
        break;
      }
      bulkCopyDimensionIdx = i;
      bulkCopySize *= dim.numElements();
    }
    if (bulkCopyDimensionIdx < 0) {
      throw new IllegalArgumentException("This array cannot be copied by bulk, since its last dimension is segmented");
    }
    return new BulkDataTransfer<>(array, bulkCopyDimensionIdx, bulkCopySize);
  }
  
  void execute(BiConsumer<BulkDataTransfer<R>, R> bulkCopy) {
    execute(bulkCopy, array, 0);
  }

  long bulkCopySize() {
    return bulkCopySize;
  }

  long totalCopied() {
    return totalCopied;
  }

  private final R array;  // The array we want to copy in bulk
  private final int bulkCopyDimensionIdx;  // The first dimension of this array that can be copied in bulk
  private final long bulkCopySize;  // The number of values that can be copied in a single bulk copy
  private long totalCopied = 0L; // The number of values copied to far

  private BulkDataTransfer(R array, int bulkCopyDimensionIdx, long bulkCopySize) {
    this.array = array;
    this.bulkCopyDimensionIdx = bulkCopyDimensionIdx;
    this.bulkCopySize = bulkCopySize;
  }

  private void execute(BiConsumer<BulkDataTransfer<R>, R> bulkCopy, R element, int dimensionIdx) {
    if (dimensionIdx == bulkCopyDimensionIdx) {
      bulkCopy.accept(this, element);
      this.totalCopied += bulkCopySize;
    } else {
      element.elements().forEach(e -> execute(bulkCopy, (R) e, dimensionIdx + 1));
    }
  }
}
