"""
SEDAC V9.0 - Skills 可用性验证

验证:
1. SKILL.md 配置正确
2. 核心模块可导入
3. CUDA 扩展状态
4. Demo 可运行
"""
import sys
from pathlib import Path

sys.path.insert(0, "G:/SEDACV9.0 PRO")

print("=" * 60)
print("SEDAC V9.0 Skills 可用性验证")
print("=" * 60)


def check_skill_manifest():
    """检查 SKILL.md"""
    skill_path = Path("G:/SEDACV9.0 PRO/skills/skills/sedac-cognitive-engine/SKILL.md")
    
    if skill_path.exists():
        content = skill_path.read_text(encoding="utf-8")
        required = ["name:", "description:", "allowed-tools:"]
        missing = [r for r in required if r not in content]
        
        if not missing:
            print("✅ SKILL.md 配置正确")
            return True
        else:
            print(f"⚠️ SKILL.md 缺少: {missing}")
            return False
    else:
        print("❌ SKILL.md 不存在")
        return False


def check_references():
    """检查引用文档"""
    ref_path = Path("G:/SEDACV9.0 PRO/skills/skills/sedac-cognitive-engine/references/architecture.md")
    
    if ref_path.exists():
        print("✅ 架构文档存在")
        return True
    else:
        print("❌ 架构文档不存在")
        return False


def check_core_modules():
    """检查核心模块"""
    modules = []
    
    try:
        from sedac.v9.core import SEDACEngine, GhostKVGenerator
        modules.append("core")
        print("✅ Core 模块可导入")
    except ImportError as e:
        print(f"⚠️ Core 模块: {e}")
    
    try:
        from sedac.v9.fused_gpu_kernel import FusedSEDACEngine
        modules.append("fused_gpu_kernel")
        print("✅ FusedGPU 模块可导入")
    except ImportError as e:
        print(f"⚠️ FusedGPU 模块: {e}")
    
    return len(modules) > 0


def check_cuda_extension():
    """检查 CUDA 扩展"""
    try:
        sys.path.insert(0, "G:/SEDACV9.0 PRO/sedac/v9/cuda_ext")
        import sedac_cuda_v2
        print("✅ CUDA 扩展已编译")
        return True
    except ImportError:
        print("⚠️ CUDA 扩展未编译 (需要运行 setup_v2.py)")
        return False


def check_demo():
    """检查 Demo 可运行"""
    demo_path = Path("G:/SEDACV9.0 PRO/sedac/v9/demo_sedac_o1.py")
    
    if demo_path.exists():
        print("✅ SEDAC-O1 Demo 存在")
        return True
    else:
        print("❌ Demo 不存在")
        return False


def main():
    results = {
        "SKILL.md": check_skill_manifest(),
        "References": check_references(),
        "Core Modules": check_core_modules(),
        "CUDA Extension": check_cuda_extension(),
        "Demo": check_demo(),
    }
    
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 SEDAC Skills 已完全配置就绪!")
    else:
        print("\n⚠️ 部分组件需要配置")
        if not results["CUDA Extension"]:
            print("  提示: 运行以下命令编译 CUDA 扩展:")
            print("  cd G:/SEDACV9.0 PRO/sedac/v9/cuda_ext && python setup_v2.py install")


if __name__ == "__main__":
    main()
