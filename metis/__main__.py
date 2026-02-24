"""
METIS CLI Entry Point
Usage: python -m metis
"""
import sys
import argparse

__version__ = "10.0.1-alpha"


def print_metis_banner() -> None:
    GREEN = "\033[0;32m"
    CYAN = "\033[0;36m"
    YELLOW = "\033[1;33m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    banner = f"""{GREEN}
███╗   ███╗███████╗████████╗██╗███████╗
████╗ ████║██╔════╝╚══██╔══╝██║██╔════╝
██╔████╔██║█████╗     ██║   ██║███████╗
██║╚██╔╝██║██╔══╝     ██║   ██║╚════██║
██║ ╚═╝ ██║███████╗   ██║   ██║███████║
╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝╚══════╝
{RESET}
 {BOLD}[SYSTEM::METIS]{RESET} {CYAN}v{__version__}{RESET}
 {DIM}Metacognitive Entropy-driven Thinking & Introspection System{RESET}
 {DIM}Named after Μῆτις — Greek Titaness of wisdom and deep thought{RESET}

 > COGNITIVE_LAYER.......[{GREEN}ONLINE{RESET}]
 > ENTROPY_MONITOR.......[{GREEN}ACTIVE{RESET}]
 > BOUNDARY_GUARD........[{GREEN}ARMED{RESET}]
 > CURIOSITY_DRIVER......[{GREEN}LISTENING{RESET}]
 > METACOGNITIVE_CORE....[{GREEN}READY{RESET}]
 > SYSTEM_2_STATUS.......[{YELLOW}STANDBY{RESET}]
"""
    print(banner)
    print(f" root@agi:~$ {GREEN}Initializing Metacognitive Core...{RESET}\n")


def cmd_info() -> None:
    """Display system information."""
    from metis import __version__ as pkg_version
    print_metis_banner()
    print(f"  Package version : {pkg_version}")
    print(f"  CLI version     : {__version__}")

    try:
        import torch
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "CPU only"
        vram = f"{torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB" if cuda else "N/A"
        print(f"  PyTorch         : {torch.__version__}")
        print(f"  CUDA            : {'✓' if cuda else '✗'} ({device})")
        print(f"  VRAM            : {vram}")
    except ImportError:
        print("  PyTorch         : not installed")

    try:
        import transformers
        print(f"  Transformers    : {transformers.__version__}")
    except ImportError:
        print("  Transformers    : not installed")

    print()


def cmd_attach(args: argparse.Namespace) -> None:
    """Attach METIS to a model and start interactive session."""
    print_metis_banner()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = args.model
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Loading model: {model_name}")
    print(f"  Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()

    from metis import Metis, MetisInference

    metis = Metis.attach(model, tokenizer)
    engine = MetisInference(metis)

    GREEN = "\033[0;32m"
    CYAN = "\033[0;36m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    print(f"  {GREEN}✓ METIS attached successfully.{RESET}")
    print(f"  {DIM}Type your prompt, or /quit to exit.{RESET}\n")

    while True:
        try:
            prompt = input(f"  {CYAN}metis>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("/quit", "/exit", "quit", "exit"):
            print("  Goodbye.")
            break

        max_tokens = args.max_tokens
        use_thinking = args.thinking

        result = engine.generate(
            prompt,
            max_tokens=max_tokens,
            use_thinking_protocol=use_thinking,
        )

        print(f"\n  {result.text}\n")
        print(f"  {DIM}[tokens={result.tokens_generated} "
              f"entropy={result.avg_token_entropy:.3f} "
              f"confidence={result.avg_confidence:.1%} "
              f"system2={result.system2_ratio:.1%} "
              f"hedged={result.was_hedged} "
              f"refused={result.was_refused}]{RESET}\n")


def cmd_experiment(args: argparse.Namespace) -> None:
    """Run the full METIS training experiment."""
    print_metis_banner()
    import subprocess
    cmd = [
        sys.executable, "run_experiment.py",
        "--model", args.model,
        "--n-prompts", str(args.n_prompts),
        "--n-samples", str(args.n_samples),
        "--max-tokens", str(args.max_tokens),
        "--output", args.output,
    ]
    subprocess.run(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="metis",
        description="METIS — Metacognitive Entropy-driven Thinking & Introspection System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # metis info
    subparsers.add_parser("info", help="Display system information and diagnostics")

    # metis attach
    p_attach = subparsers.add_parser("attach", help="Attach METIS to a model (interactive)")
    p_attach.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                          help="HuggingFace model name")
    p_attach.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    p_attach.add_argument("--max-tokens", type=int, default=512, help="Max generation tokens")
    p_attach.add_argument("--thinking", action="store_true", help="Enable Thinking Protocol")

    # metis experiment
    p_exp = subparsers.add_parser("experiment", help="Run full METIS training experiment")
    p_exp.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p_exp.add_argument("--n-prompts", type=int, default=300)
    p_exp.add_argument("--n-samples", type=int, default=8)
    p_exp.add_argument("--max-tokens", type=int, default=512)
    p_exp.add_argument("--output", type=str, default="./experiment_full")

    args = parser.parse_args()

    if args.command == "info":
        cmd_info()
    elif args.command == "attach":
        cmd_attach(args)
    elif args.command == "experiment":
        cmd_experiment(args)
    else:
        print_metis_banner()
        parser.print_help()


if __name__ == "__main__":
    main()
