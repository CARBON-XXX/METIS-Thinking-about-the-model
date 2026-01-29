# SEDAC V9.0 Production Dockerfile
# 基于 NVIDIA NGC PyTorch 镜像

FROM nvcr.io/nvidia/pytorch:24.01-py3

LABEL maintainer="SEDAC Team"
LABEL version="9.0.0"
LABEL description="SEDAC V9.0 Production Inference Server"

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV SEDAC_HOME=/app/sedac

# 工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 复制 SEDAC 代码
COPY sedac/ /app/sedac/

# 编译 CUDA 扩展
WORKDIR /app/sedac/v9/cuda_ext
RUN python setup_v2.py install || echo "CUDA extension build skipped"

WORKDIR /app

# 暴露端口
EXPOSE 8000
EXPOSE 9090

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
ENTRYPOINT ["python", "-m", "sedac.v9.production.server"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
