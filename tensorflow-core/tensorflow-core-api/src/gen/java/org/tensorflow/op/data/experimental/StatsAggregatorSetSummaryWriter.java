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

import org.tensorflow.Operand;
import org.tensorflow.Operation;
import org.tensorflow.OperationBuilder;
import org.tensorflow.op.PrimitiveOp;
import org.tensorflow.op.Scope;
import org.tensorflow.op.annotation.Endpoint;
import org.tensorflow.op.annotation.Operator;

/**
 * Set a summary_writer_interface to record statistics using given stats_aggregator.
 */
public final class StatsAggregatorSetSummaryWriter extends PrimitiveOp {
  
  /**
   * Factory method to create a class wrapping a new StatsAggregatorSetSummaryWriter operation.
   * 
   * @param scope current scope
   * @param statsAggregator 
   * @param summary 
   * @return a new instance of StatsAggregatorSetSummaryWriter
   */
  @Endpoint
  public static StatsAggregatorSetSummaryWriter create(Scope scope, Operand<?> statsAggregator, Operand<?> summary) {
    OperationBuilder opBuilder = scope.env().opBuilder("StatsAggregatorSetSummaryWriter", scope.makeOpName("StatsAggregatorSetSummaryWriter"));
    opBuilder.addInput(statsAggregator.asOutput());
    opBuilder.addInput(summary.asOutput());
    opBuilder = scope.applyControlDependencies(opBuilder);
    return new StatsAggregatorSetSummaryWriter(opBuilder.build());
  }
  
  
  private StatsAggregatorSetSummaryWriter(Operation operation) {
    super(operation);
  }
}
