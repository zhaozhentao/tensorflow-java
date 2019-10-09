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

package org.tensorflow.op.core;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;
import org.tensorflow.types.TInt64;

/**
 * Outputs random integers from a uniform distribution.
 * <p>
 * The generated values are uniform integers covering the whole range of `dtype`. 
 * 
 * @param <U> data type for {@code output()} output
 */
public final class StatefulUniformFullInt<U> extends PrimitiveOp implements Operand<U> {
  
  /**
   * Factory method to create a class wrapping a new StatefulUniformFullInt operation.
   * 
   * @param scope current scope
   * @param resource The handle of the resource variable that stores the state of the RNG.
   * @param algorithm The RNG algorithm.
   * @param shape The shape of the output tensor.
   * @param dtype The type of the output.
   * @return a new instance of StatefulUniformFullInt
   */
  public static <U, T> StatefulUniformFullInt<U> create(Scope scope, Operand<?> resource, Operand<TInt64> algorithm, Operand<T> shape, DataType<U> dtype) {
    OperationBuilder opBuilder = scope.env().opBuilder("StatefulUniformFullInt", scope.makeOpName("StatefulUniformFullInt"));
    opBuilder.addInput(resource.asOutput());
    opBuilder.addInput(algorithm.asOutput());
    opBuilder.addInput(shape.asOutput());
    opBuilder = scope.applyControlDependencies(opBuilder);
    opBuilder.setAttr("dtype", dtype);
    return new StatefulUniformFullInt<U>(opBuilder.build());
  }
  
  /**
   * Random values with specified shape.
   */
  public Output<U> output() {
    return output;
  }
  
  @Override
  public Output<U> asOutput() {
    return output;
  }
  
  private Output<U> output;
  
  private StatefulUniformFullInt(Operation operation) {
    super(operation);
    int outputIdx = 0;
    output = operation.output(outputIdx++);
  }
}
