/**
 * WebGPU ECG Renderer — native WGSL shader pipeline.
 *
 * Manages GPU device, buffers, and render loop for real-time
 * entropy waveform visualization.
 */

import { ECG_SHADER } from "./shaders";

/** Max samples in the ring buffer */
const BUFFER_SIZE = 1024;

/** Uniform buffer layout (must match WGSL struct) */
interface Uniforms {
  viewport: [number, number];
  sampleCount: number;
  bufferSize: number;
  time: number;
  lineWidth: number;
  fastThreshold: number;
  deepThreshold: number;
}

export class ECGRenderer {
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private pipeline: GPURenderPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private entropyBuffer: GPUBuffer | null = null;
  private decisionBuffer: GPUBuffer | null = null;
  private bindGroup: GPUBindGroup | null = null;
  private format: GPUTextureFormat = "bgra8unorm";

  private entropyData: Float32Array = new Float32Array(BUFFER_SIZE);
  private decisionData: Uint32Array = new Uint32Array(BUFFER_SIZE);
  private writeHead = 0;
  private sampleCount = 0;
  private startTime = 0;
  private animFrameId = 0;
  private disposed = false;

  private fastThreshold = 1.5;
  private deepThreshold = 2.0;

  async init(canvas: HTMLCanvasElement): Promise<boolean> {
    if (!navigator.gpu) {
      console.warn("WebGPU not supported — falling back to Canvas2D");
      return false;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;

    this.device = await adapter.requestDevice();
    this.context = canvas.getContext("webgpu") as GPUCanvasContext;
    this.format = navigator.gpu.getPreferredCanvasFormat();

    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: "premultiplied",
    });

    this.createBuffers();
    this.createPipeline();
    this.startTime = performance.now() / 1000;

    return true;
  }

  private createBuffers(): void {
    const dev = this.device!;

    // Uniform buffer: 8 floats = 32 bytes
    this.uniformBuffer = dev.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Entropy signal buffer
    this.entropyBuffer = dev.createBuffer({
      size: BUFFER_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // Decision buffer (0=FAST, 1=NORMAL, 2=DEEP)
    this.decisionBuffer = dev.createBuffer({
      size: BUFFER_SIZE * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  private createPipeline(): void {
    const dev = this.device!;

    const shaderModule = dev.createShaderModule({ code: ECG_SHADER });

    const bindGroupLayout = dev.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ],
    });

    this.bindGroup = dev.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer! } },
        { binding: 1, resource: { buffer: this.entropyBuffer! } },
        { binding: 2, resource: { buffer: this.decisionBuffer! } },
      ],
    });

    this.pipeline = dev.createRenderPipeline({
      layout: dev.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout],
      }),
      vertex: {
        module: shaderModule,
        entryPoint: "vs_main",
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs_main",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  /** Push a new signal sample */
  pushSample(entropy: number, decision: number): void {
    this.entropyData[this.writeHead] = entropy;
    this.decisionData[this.writeHead] = decision;
    this.writeHead = (this.writeHead + 1) % BUFFER_SIZE;
    if (this.sampleCount < BUFFER_SIZE) {
      this.sampleCount++;
    }
  }

  /** Update thresholds from controller stats */
  setThresholds(fast: number, deep: number): void {
    this.fastThreshold = fast;
    this.deepThreshold = deep;
  }

  /** Start the render loop */
  startLoop(): void {
    this.disposed = false;
    const loop = (): void => {
      if (this.disposed) return;
      this.render();
      this.animFrameId = requestAnimationFrame(loop);
    };
    this.animFrameId = requestAnimationFrame(loop);
  }

  /** Stop the render loop */
  stopLoop(): void {
    this.disposed = true;
    cancelAnimationFrame(this.animFrameId);
  }

  private render(): void {
    if (!this.device || !this.context || !this.pipeline || !this.bindGroup) return;

    const canvas = this.context.canvas as HTMLCanvasElement;
    const width = canvas.width;
    const height = canvas.height;

    // Reorder ring buffer into linear array for GPU
    const linearEntropy = new Float32Array(this.sampleCount);
    const linearDecision = new Uint32Array(this.sampleCount);
    for (let i = 0; i < this.sampleCount; i++) {
      const idx =
        this.sampleCount < BUFFER_SIZE
          ? i
          : (this.writeHead + i) % BUFFER_SIZE;
      linearEntropy[i] = this.entropyData[idx];
      linearDecision[i] = this.decisionData[idx];
    }

    // Upload buffers
    const uniformData = new Float32Array([
      width,
      height,
      this.sampleCount,
      BUFFER_SIZE,
      performance.now() / 1000 - this.startTime,
      2.0, // line width in pixels
      this.fastThreshold,
      this.deepThreshold,
    ]);

    this.device.queue.writeBuffer(this.uniformBuffer!, 0, uniformData);
    if (this.sampleCount > 0) {
      this.device.queue.writeBuffer(this.entropyBuffer!, 0, linearEntropy);
      this.device.queue.writeBuffer(this.decisionBuffer!, 0, linearDecision);
    }

    // Render
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.04, g: 0.06, b: 0.1, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });

    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3); // Fullscreen triangle
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }

  /** Clean up GPU resources */
  dispose(): void {
    this.stopLoop();
    this.uniformBuffer?.destroy();
    this.entropyBuffer?.destroy();
    this.decisionBuffer?.destroy();
    this.device?.destroy();
    this.device = null;
  }
}
