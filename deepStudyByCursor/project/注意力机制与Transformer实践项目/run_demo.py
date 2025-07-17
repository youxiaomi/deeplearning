#!/usr/bin/env python3
"""
注意力机制与Transformer实践项目演示脚本
Attention Mechanism and Transformer Practice Project Demo Script

运行这个脚本来体验完整的项目演示
Run this script to experience the complete project demonstration
"""

import sys
import os
import traceback

def main():
    """
    主演示函数
    Main demonstration function
    """
    print("=" * 60)
    print("    注意力机制与Transformer实践项目演示")
    print("    Attention Mechanism and Transformer Practice Project Demo")
    print("=" * 60)
    print()
    
    demos = [
        ("基础注意力机制演示", "Basic Attention Mechanism Demo", "attention_mechanism", "demo_basic_attention"),
        ("完整Transformer编码器演示", "Complete Transformer Encoder Demo", "attention_mechanism", "demo_transformer_encoder"),
        ("注意力可视化演示", "Attention Visualization Demo", "attention_mechanism", "demo_attention_visualization"),
        ("情感分析项目演示", "Sentiment Analysis Project Demo", "sentiment_analysis_project", "main"),
        ("文本生成项目演示", "Text Generation Project Demo", "text_generation_project", "demo_text_generation")
    ]
    
    print("可用的演示 (Available Demonstrations):")
    for i, (name_zh, name_en, module, func) in enumerate(demos, 1):
        print(f"{i}. {name_zh} ({name_en})")
    print("0. 运行所有演示 (Run All Demonstrations)")
    print()
    
    try:
        choice = input("请选择要运行的演示 (Please select a demo to run): ")
        choice = int(choice)
        
        if choice == 0:
            # 运行所有演示
            # Run all demonstrations
            print("\n开始运行所有演示...\n")
            for i, (name_zh, name_en, module, func) in enumerate(demos, 1):
                print(f"\n{'='*50}")
                print(f"演示 {i}: {name_zh}")
                print(f"Demo {i}: {name_en}")
                print('='*50)
                run_demo(module, func)
                
        elif 1 <= choice <= len(demos):
            # 运行指定演示
            # Run specific demonstration
            name_zh, name_en, module, func = demos[choice - 1]
            print(f"\n运行演示: {name_zh}")
            print(f"Running demo: {name_en}\n")
            run_demo(module, func)
            
        else:
            print("无效的选择 (Invalid choice)")
            
    except ValueError:
        print("请输入有效的数字 (Please enter a valid number)")
    except KeyboardInterrupt:
        print("\n\n用户中断演示 (User interrupted demonstration)")
    except Exception as e:
        print(f"\n演示过程中出现错误 (Error during demonstration): {e}")
        traceback.print_exc()


def run_demo(module_name, function_name):
    """
    运行指定的演示
    Run specific demonstration
    
    Args:
        module_name: 模块名称
        function_name: 函数名称
    """
    try:
        # 动态导入模块
        # Dynamic module import
        if module_name == "attention_mechanism":
            from attention_mechanism import demo_basic_attention, demo_transformer_encoder, demo_attention_visualization
            
            if function_name == "demo_basic_attention":
                demo_basic_attention()
            elif function_name == "demo_transformer_encoder":
                demo_transformer_encoder()
            elif function_name == "demo_attention_visualization":
                demo_attention_visualization()
                
        elif module_name == "sentiment_analysis_project":
            from sentiment_analysis_project import main
            main()
            
        elif module_name == "text_generation_project":
            from text_generation_project import demo_text_generation
            demo_text_generation()
            
        print(f"\n✅ 演示完成! (Demonstration completed!)")
        
    except ImportError as e:
        print(f"❌ 导入错误 (Import error): {e}")
        print("请确保所有依赖都已安装 (Please ensure all dependencies are installed)")
        print("运行: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ 运行错误 (Runtime error): {e}")
        traceback.print_exc()


def check_dependencies():
    """
    检查依赖是否已安装
    Check if dependencies are installed
    """
    required_packages = ['torch', 'matplotlib', 'seaborn', 'numpy', 'jieba', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖包 (Missing required packages):")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装依赖 (Please run the following command to install dependencies):")
        print("pip install -r requirements.txt")
        return False
    
    return True


def setup_environment():
    """
    设置环境
    Setup environment
    """
    # 设置中文分词
    # Setup Chinese tokenization
    try:
        import jieba
        jieba.setLogLevel(jieba.logging.INFO)  # 减少日志输出
    except ImportError:
        pass
    
    # 设置matplotlib后端（如果在服务器环境）
    # Setup matplotlib backend (if in server environment)
    try:
        import matplotlib
        matplotlib.use('Agg')  # 如果没有GUI，使用Agg后端
    except ImportError:
        pass


if __name__ == "__main__":
    print("正在检查环境和依赖... (Checking environment and dependencies...)")
    
    # 设置环境
    # Setup environment
    setup_environment()
    
    # 检查依赖
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ 环境检查通过! (Environment check passed!)\n")
    
    # 运行主程序
    # Run main program
    main()