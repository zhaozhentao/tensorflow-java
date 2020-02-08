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

package org.tensorflow.op.data.experimental;

import java.util.List;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Output;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;
import org.tensorflow.op.annotation.Endpoint;
import org.tensorflow.op.annotation.Operator;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.family.TType;

/**
 * Creates a dataset that contains the unique elements of `input_dataset`.
 */
public final class UniqueDataset extends PrimitiveOp implements Operand<TType> {
  
  /**
   * Factory method to create a class wrapping a new UniqueDataset operation.
   * 
   * @param scope current scope
   * @param inputDataset 
   * @param outputTypes 
   * @param outputShapes 
   * @return a new instance of UniqueDataset
   */
  @Endpoint
  public static UniqueDataset create(Scope scope, Operand<?> inputDataset, List<DataType<?>> outputTypes, List<Shape> outputShapes) {
    OperationBuilder opBuilder = scope.env().opBuilder("ExperimentalUniqueDataset", scope.makeOpName("UniqueDataset"));
    opBuilder.addInput(inputDataset.asOutput());
    opBuilder = scope.applyControlDependencies(opBuilder);
    DataType[] outputTypesArray = new DataType[outputTypes.size()];
    for (int i = 0; i < outputTypesArray.length; ++i) {
      outputTypesArray[i] = outputTypes.get(i);
    }
    opBuilder.setAttr("output_types", outputTypesArray);
    Shape[] outputShapesArray = new Shape[outputShapes.size()];
    for (int i = 0; i < outputShapesArray.length; ++i) {
      outputShapesArray[i] = outputShapes.get(i);
    }
    opBuilder.setAttr("output_shapes", outputShapesArray);
    return new UniqueDataset(opBuilder.build());
  }
  
  /**
   */
  public Output<?> handle() {
    return handle;
  }
  
  @Override
  @SuppressWarnings("unchecked")
  public Output<TType> asOutput() {
    return (Output<TType>) handle;
  }
  
  private Output<?> handle;
  
  private UniqueDataset(Operation operation) {
    super(operation);
    int outputIdx = 0;
    handle = operation.output(outputIdx++);
  }
}
