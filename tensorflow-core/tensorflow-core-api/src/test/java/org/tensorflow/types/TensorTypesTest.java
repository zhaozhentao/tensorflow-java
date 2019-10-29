package org.tensorflow.types;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import org.junit.Test;
import org.tensorflow.EagerSession;
import org.tensorflow.Tensor;
import org.tensorflow.nio.nd.IntNdArray;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.math.Sub;

import static org.tensorflow.nio.StaticApi.*;

public class TensorTypesTest {

  @Test
  public void initializeTensorsWithZeros() {
    // Allocate a tensor of 32-bits integer of the shape (2, 3, 2)
    Tensor<TInt32> tensor = TInt32.tensorOfShape(2, 3, 2);

    // Access tensor memory directly
    IntNdArray tensorData = tensor.data();
    assertEquals(3, tensorData.rank());
    assertEquals(12, tensorData.size());

    try (EagerSession session = EagerSession.create()) {
      Ops tf = Ops.create(session);

      // Initialize tensor memory with zeros
      tensorData.write(bufferOfInts(tensorData.size()));
      tensorData.scalars().forEach(scalar ->
          assertEquals(0, scalar.getInt())
      );
      Constant<TInt32> x = tf.constant(tensor);  // take snapshot of `tensor` with all zeros

      // Initialize tensor memory with all ones
      int[] ones = new int[(int)tensorData.size()];
      Arrays.fill(ones, 1);
      tensorData.write(ones);
      tensorData.scalars().forEach(scalar ->
          assertEquals(1, scalar.getInt())
      );
      Constant<TInt32> y = tf.constant(tensor);  // take snapshot of `tensor` with all ones

      // Subtract y from x and validate the result
      Sub<TInt32> sub = tf.math.sub(x, y);
      sub.tensorData().scalars().forEach(scalar ->
          assertEquals(-1, scalar.getInt())
      );
    }
  }
}
