#!/usr/bin/env python3
"""
SEDAC V9.0 Production Runner

统一入口脚本，支持多种运行模式
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sedac")


def cmd_serve(args):
    """启动 API 服务器"""
    from .server import create_server
    from .config import ProductionConfig
    
    config = None
    if args.config:
        import yaml
        with open(args.config) as f:
            config_data = yaml.safe_load(f)
        config = ProductionConfig._from_dict(config_data)
    
    server = create_server(
        model_name=args.model,
        config=config,
        host=args.host,
        port=args.port,
    )
    
    logger.info(f"Starting SEDAC server on {args.host}:{args.port}")
    server.run()


def cmd_benchmark(args):
    """运行基准测试"""
    from .benchmark import ProductionBenchmark, BenchmarkConfig
    
    config = BenchmarkConfig(
        batch_sizes=[int(b) for b in args.batch_sizes.split(",")],
        input_lengths=[int(l) for l in args.input_lengths.split(",")],
        output_lengths=[int(l) for l in args.output_lengths.split(",")],
        warmup_iterations=args.warmup,
        benchmark_iterations=args.iterations,
        include_baseline=not args.sedac_only,
        output_file=args.output,
    )
    
    benchmark = ProductionBenchmark(args.model)
    report = benchmark.run(config)
    report.print_summary()


def cmd_train(args):
    """训练 Ghost KV"""
    from .trainer import GhostKVTrainer, TrainingConfig
    from .config import ProductionConfig
    from .engine import GhostKVGenerator
    
    logger.info("Loading training data...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")
        texts = [item["text"] for item in dataset if len(item["text"]) > 100][:500]
    except:
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 10
            for _ in range(100)
        ]
    
    logger.info(f"Loaded {len(texts)} training samples")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    logger.info(f"Loading teacher model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    
    prod_config = ProductionConfig()
    try:
        from .config import ModelConfig
        prod_config.model = ModelConfig.from_pretrained(args.model)
    except:
        pass
    
    train_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
    
    ghost_kv = GhostKVGenerator(prod_config)
    
    from .trainer import KVDistillationDataset, collate_kv_samples
    from torch.utils.data import DataLoader
    
    dataset = KVDistillationDataset(model, tokenizer, texts, prod_config)
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=collate_kv_samples,
    )
    
    trainer = GhostKVTrainer(ghost_kv, train_config, prod_config)
    results = trainer.train(loader)
    
    logger.info(f"Training complete. Best similarity: {results['best_similarity']:.4f}")


def cmd_test(args):
    """运行测试"""
    from .tests import run_tests
    
    result = run_tests(verbosity=args.verbosity)
    
    if result.wasSuccessful():
        logger.info("All tests passed!")
        sys.exit(0)
    else:
        logger.error(f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors")
        sys.exit(1)


def cmd_infer(args):
    """单次推理"""
    from .inference import create_pipeline, GenerationConfig
    
    logger.info(f"Loading model: {args.model}")
    pipeline = create_pipeline(args.model)
    
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = input("Enter prompt: ")
    
    logger.info("Generating...")
    result = pipeline(prompt, gen_config)
    
    print("\n" + "=" * 60)
    print("Generated Text:")
    print("=" * 60)
    print(result.generated_text)
    print("\n" + "-" * 60)
    print(f"Tokens: {result.generated_tokens}")
    print(f"Latency: {result.total_latency_ms:.2f}ms")
    print(f"TPS: {result.tokens_per_second:.1f}")
    print(f"Avg Exit Layer: {result.avg_exit_layer:.1f}")
    print(f"Skip Ratio: {result.skip_ratio*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="SEDAC V9.0 Production CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # serve
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--config", type=str, help="Config YAML file")
    serve_parser.set_defaults(func=cmd_serve)
    
    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    bench_parser.add_argument("--batch-sizes", type=str, default="1,4")
    bench_parser.add_argument("--input-lengths", type=str, default="128,256")
    bench_parser.add_argument("--output-lengths", type=str, default="64,128")
    bench_parser.add_argument("--warmup", type=int, default=3)
    bench_parser.add_argument("--iterations", type=int, default=10)
    bench_parser.add_argument("--sedac-only", action="store_true")
    bench_parser.add_argument("--output", type=str, default="benchmark_report.json")
    bench_parser.set_defaults(func=cmd_benchmark)
    
    # train
    train_parser = subparsers.add_parser("train", help="Train Ghost KV")
    train_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--output-dir", type=str, default="./ghost_kv_checkpoints")
    train_parser.set_defaults(func=cmd_train)
    
    # test
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--verbosity", type=int, default=2)
    test_parser.set_defaults(func=cmd_test)
    
    # infer
    infer_parser = subparsers.add_parser("infer", help="Single inference")
    infer_parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    infer_parser.add_argument("--prompt", type=str, help="Input prompt")
    infer_parser.add_argument("--max-tokens", type=int, default=256)
    infer_parser.add_argument("--temperature", type=float, default=0.7)
    infer_parser.add_argument("--top-p", type=float, default=0.9)
    infer_parser.set_defaults(func=cmd_infer)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
