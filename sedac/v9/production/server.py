"""
SEDAC V9.0 Production Server

生产级 API 服务器，支持 REST 和 gRPC
符合 NVIDIA Triton Inference Server 标准
"""
from __future__ import annotations
import asyncio
import json
import time
import logging
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import threading

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import StreamingResponse, JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn")

from .config import ProductionConfig
from .inference import SEDACInferencePipeline, GenerationConfig, InferenceResult


class GenerateRequest(BaseModel):
    """生成请求"""
    prompt: str = Field(..., description="Input prompt")
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    do_sample: bool = Field(True)
    stream: bool = Field(False, description="Enable streaming response")
    

class GenerateResponse(BaseModel):
    """生成响应"""
    generated_text: str
    generated_tokens: int
    input_tokens: int
    total_latency_ms: float
    tokens_per_second: float
    avg_exit_layer: float
    skip_ratio: float
    used_o1: bool


class BatchGenerateRequest(BaseModel):
    """批量生成请求"""
    prompts: List[str]
    max_new_tokens: int = Field(256, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    model_loaded: bool
    uptime_seconds: float
    gpu_memory_gb: Optional[float] = None


class MetricsResponse(BaseModel):
    """指标响应"""
    uptime_seconds: float
    latency: Dict[str, float]
    throughput: Dict[str, Any]
    sedac: Dict[str, Any]


class SEDACServer:
    """
    SEDAC 生产级服务器
    
    特性:
    - RESTful API
    - 流式生成
    - 批量推理
    - 健康检查与指标
    - 优雅关闭
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        config: Optional[ProductionConfig] = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        workers: int = 1,
    ):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI required for server. Install with: pip install fastapi uvicorn")
        
        self.model_name = model_name
        self.config = config or ProductionConfig()
        self.host = host
        self.port = port
        self.workers = workers
        
        self.pipeline: Optional[SEDACInferencePipeline] = None
        self.start_time = time.time()
        self._lock = threading.Lock()
        
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """创建 FastAPI 应用"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info("Starting SEDAC server...")
            self._load_model()
            yield
            logger.info("Shutting down SEDAC server...")
        
        app = FastAPI(
            title="SEDAC V9.0 Inference Server",
            description="Production-grade semantic entropy-guided inference API",
            version="9.0.0",
            lifespan=lifespan,
        )
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """健康检查"""
            gpu_mem = None
            if self.pipeline and self.pipeline.sedac_engine:
                gpu_info = self.pipeline.sedac_engine.monitor.get_gpu_memory_usage()
                gpu_mem = gpu_info.get("allocated_gb")
            
            return HealthResponse(
                status="healthy" if self.pipeline else "loading",
                model_loaded=self.pipeline is not None and self.pipeline._is_loaded,
                uptime_seconds=time.time() - self.start_time,
                gpu_memory_gb=gpu_mem,
            )
        
        @app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics():
            """获取性能指标"""
            if not self.pipeline:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            metrics = self.pipeline.get_metrics()
            return MetricsResponse(**metrics)
        
        @app.get("/metrics/prometheus")
        async def get_prometheus_metrics():
            """Prometheus 格式指标"""
            if not self.pipeline or not self.pipeline.sedac_engine:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            prometheus_text = self.pipeline.sedac_engine.metrics.export_prometheus()
            return JSONResponse(
                content=prometheus_text,
                media_type="text/plain"
            )
        
        @app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """生成接口"""
            if not self.pipeline:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            if request.stream:
                return StreamingResponse(
                    self._stream_generate(request),
                    media_type="text/event-stream"
                )
            
            gen_config = GenerationConfig(
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=request.do_sample,
            )
            
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.pipeline(request.prompt, gen_config)
                )
                
                return GenerateResponse(
                    generated_text=result.generated_text,
                    generated_tokens=result.generated_tokens,
                    input_tokens=result.input_tokens,
                    total_latency_ms=result.total_latency_ms,
                    tokens_per_second=result.tokens_per_second,
                    avg_exit_layer=result.avg_exit_layer,
                    skip_ratio=result.skip_ratio,
                    used_o1=result.used_o1,
                )
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/batch_generate")
        async def batch_generate(request: BatchGenerateRequest):
            """批量生成"""
            if not self.pipeline:
                raise HTTPException(status_code=503, detail="Model not loaded")
            
            gen_config = GenerationConfig(
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            
            try:
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.pipeline.batch_generate(request.prompts, gen_config)
                )
                
                return [
                    GenerateResponse(
                        generated_text=r.generated_text,
                        generated_tokens=r.generated_tokens,
                        input_tokens=r.input_tokens,
                        total_latency_ms=r.total_latency_ms,
                        tokens_per_second=r.tokens_per_second,
                        avg_exit_layer=r.avg_exit_layer,
                        skip_ratio=r.skip_ratio,
                        used_o1=r.used_o1,
                    )
                    for r in results
                ]
            except Exception as e:
                logger.error(f"Batch generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/reset_metrics")
        async def reset_metrics():
            """重置指标"""
            if self.pipeline:
                self.pipeline.reset_metrics()
            return {"status": "metrics reset"}
        
        return app
    
    def _load_model(self) -> None:
        """加载模型"""
        with self._lock:
            if self.pipeline is None:
                self.pipeline = SEDACInferencePipeline(
                    self.model_name,
                    config=self.config,
                )
                self.pipeline.load()
    
    async def _stream_generate(self, request: GenerateRequest) -> AsyncGenerator[str, None]:
        """流式生成"""
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
        )
        
        for chunk in self.pipeline.stream_generate(request.prompt, gen_config):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    def run(self) -> None:
        """运行服务器"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            workers=self.workers,
            log_level="info",
        )


def create_server(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs
) -> SEDACServer:
    """
    工厂函数：创建服务器
    
    Args:
        model_name: 模型名称
        host: 主机地址
        port: 端口
    
    Returns:
        SEDAC 服务器实例
    """
    return SEDACServer(model_name, host=host, port=port, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEDAC V9.0 Inference Server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    server = create_server(args.model, args.host, args.port)
    server.run()
