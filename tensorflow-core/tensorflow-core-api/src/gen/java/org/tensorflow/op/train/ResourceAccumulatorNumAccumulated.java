/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/

// This class has been generated, DO NOT EDIT!

package org.tensorflow.op.train;

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;
import org.tensorflow.op.annotation.Endpoint;
import org.tensorflow.op.annotation.Operator;
import org.tensorflow.types.TInt32;

/**
 * Returns the number of gradients aggregated in the given accumulators.
 */
public final class ResourceAccumulatorNumAccumulated extends PrimitiveOp implements Operand<TInt32> {
  
  /**
   * Factory method to create a class wrapping a new ResourceAccumulatorNumAccumulated operation.
   * 
   * @param scope current scope
   * @param handle The handle to an accumulator.
   * @return a new instance of ResourceAccumulatorNumAccumulated
   */
  @Endpoint
  public static ResourceAccumulatorNumAccumulated create(Scope scope, Operand<?> handle) {
    OperationBuilder opBuilder = scope.env().opBuilder("ResourceAccumulatorNumAccumulated", scope.makeOpName("ResourceAccumulatorNumAccumulated"));
    opBuilder.addInput(handle.asOutput());
    opBuilder = scope.applyControlDependencies(opBuilder);
    return new ResourceAccumulatorNumAccumulated(opBuilder.build());
  }
  
  /**
   * The number of gradients aggregated in the given accumulator.
   */
  public Output<TInt32> numAccumulated() {
    return numAccumulated;
  }
  
  @Override
  public Output<TInt32> asOutput() {
    return numAccumulated;
  }
  
  private Output<TInt32> numAccumulated;
  
  private ResourceAccumulatorNumAccumulated(Operation operation) {
    super(operation);
    int outputIdx = 0;
    numAccumulated = operation.output(outputIdx++);
  }
}
