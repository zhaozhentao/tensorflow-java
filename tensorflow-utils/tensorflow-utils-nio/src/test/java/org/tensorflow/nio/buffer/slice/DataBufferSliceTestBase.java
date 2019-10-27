package org.tensorflow.nio.buffer.slice;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.tensorflow.nio.buffer.DataBufferTestBase;

abstract class DataBufferSliceTestBase<T> extends DataBufferTestBase<T> {

  @Override
  protected long maxCapacity() {
    return Long.MAX_VALUE;
  }

  @Override
  public void capacities() {
    // Slice capacities only depends on their delegate one, so it is pointless to test them here
  }

  @Test
  public void move() {
    DataBufferSlice<T> slice = allocate(100L).mutableSlice();
    assertEquals(0L, slice.position());
    assertEquals(100L, slice.limit());
    assertEquals(100L, slice.capacity());
    assertEquals(100L, slice.remaining());

    slice.moveTo(5L);
    assertEquals(0L, slice.position());
    assertEquals(95L, slice.limit());
    assertEquals(95L, slice.capacity());
    assertEquals(95L, slice.remaining());

    slice.position(20L).limit(80L);
    assertEquals(20L, slice.position());
    assertEquals(80L, slice.limit());

    slice.moveTo(10L);
    assertEquals(0L, slice.position());
    assertEquals(90L, slice.limit());
    assertEquals(90L, slice.capacity());
    assertEquals(90L, slice.remaining());

    slice.moveTo(0L, 70L);
    assertEquals(0L, slice.position());
    assertEquals(70L, slice.limit());
    assertEquals(70L, slice.capacity());
    assertEquals(70L, slice.remaining());
  }
}
