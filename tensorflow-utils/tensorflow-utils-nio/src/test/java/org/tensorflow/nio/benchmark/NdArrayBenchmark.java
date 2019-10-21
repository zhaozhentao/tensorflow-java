/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.tensorflow.nio.benchmark;

import static org.tensorflow.nio.nd.NdArrays.vector;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import javax.imageio.ImageIO;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.runner.RunnerException;
import org.tensorflow.nio.nd.FloatNdArray;
import org.tensorflow.nio.nd.NdArrays;
import org.tensorflow.nio.nd.Shape;
import org.tensorflow.nio.nd.index.Indices;

@Fork(value = 1, jvmArgs = {"-Xms4G", "-Xmx4G"})
@BenchmarkMode(Mode.AverageTime)
@Measurement(iterations = 5)
@State(Scope.Benchmark)
public class NdArrayBenchmark {

	static final String TEST_IMAGE = "1500x916.jpg";
	static final int BATCH_SIZE = 100;

	private FloatNdArray array;
	private FloatNdArray pixels;
	private FloatNdArray channels;

	@Setup
	public void setUp() throws IOException {
		BufferedImage image = ImageIO.read(getClass().getClassLoader().getResourceAsStream(TEST_IMAGE));

		int numPixels = image.getWidth() * image.getHeight();
		pixels = NdArrays.ofFloats(Shape.make(numPixels, 3));
		channels = NdArrays.ofFloats(Shape.make(3, numPixels));

		Raster imageData = image.getData();
		float[] pixel = new float[3];
		for (int y = 0, pixelIdx = 0; y < image.getHeight(); ++y) {
			for (int x = 0; x < image.getWidth(); ++x, ++pixelIdx) {
				imageData.getPixel(x, y, pixel);
				pixels.set(vector(pixel), pixelIdx);
				channels
						.setFloat(pixel[2], 0, pixelIdx)  // R
						.setFloat(pixel[1], 1, pixelIdx)  // G
						.setFloat(pixel[0], 2, pixelIdx); // B
			}
		}

		array = NdArrays.ofFloats(Shape.make(BATCH_SIZE, 3, numPixels));
	}

	@Benchmark
	public void slicing() {
		array.elements(0).forEach(batch ->
			batch.slice(Indices.all(), Indices.at(0))
		);
	}

	@Benchmark
	public void writeByChannel() {
	  array.elements(0).forEach(batch ->
			batch.set(channels)
		);
	}

	@Benchmark
	public void writeByPixelBySlice() {
		array.elements(0).forEach(batch ->
				pixels.elements(0).forEachIdx((coords, pixel) ->
            batch.slice(Indices.all(), Indices.at(coords[0])).set(pixel)
				)
		);
	}

	public static void main(String[] args) throws IOException, RunnerException {
		org.openjdk.jmh.Main.main(args);
	}
}
