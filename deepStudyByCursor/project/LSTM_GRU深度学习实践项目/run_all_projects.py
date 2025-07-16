"""
LSTM/GRU深度学习实践项目主程序
LSTM/GRU Deep Learning Practice Project Main Program

这是整个项目的主入口，可以选择运行不同的子项目。
This is the main entry point for the entire project, allowing you to choose different sub-projects to run.
"""

import os
import sys
import subprocess
import time
from typing import Dict, List


class ProjectRunner:
    """
    项目运行器
    Project Runner
    
    这个类就像一个导航员，帮你选择和运行不同的学习项目。
    This class is like a navigator that helps you choose and run different learning projects.
    """
    
    def __init__(self):
        self.projects = {
            "1": {
                "name": "基础理论实现 | Basic Theory Implementation",
                "description": "从零实现LSTM和GRU，深入理解核心算法",
                "description_en": "Implement LSTM and GRU from scratch, understand core algorithms",
                "scripts": [
                    "01_基础理论实现/lstm_from_scratch.py",
                    "01_基础理论实现/gru_from_scratch.py"
                ]
            },
            "2": {
                "name": "文本情感分析 | Text Sentiment Analysis", 
                "description": "使用LSTM/GRU进行中文文本情感分类",
                "description_en": "Use LSTM/GRU for Chinese text sentiment classification",
                "scripts": [
                    "02_文本情感分析/sentiment_lstm.py"
                ]
            },
            "3": {
                "name": "股票价格预测 | Stock Price Prediction",
                "description": "基于历史数据预测股票价格走势",
                "description_en": "Predict stock price trends based on historical data",
                "scripts": [
                    "03_股票价格预测/stock_lstm.py"
                ]
            },
            "4": {
                "name": "综合对比分析 | Comprehensive Comparison",
                "description": "全面对比LSTM和GRU的性能差异",
                "description_en": "Comprehensive comparison of LSTM and GRU performance",
                "scripts": [
                    "06_综合对比分析/model_comparison.py"
                ]
            }
        }
        
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
    
    def display_menu(self):
        """
        显示项目菜单
        Display project menu
        """
        print("🎓 欢迎来到LSTM/GRU深度学习实践项目！")
        print("🎓 Welcome to LSTM/GRU Deep Learning Practice Project!")
        print("=" * 80)
        print()
        
        print("📚 学习路径建议 | Recommended Learning Path:")
        print("   建议按照1→2→3→4的顺序进行学习，循序渐进掌握LSTM/GRU")
        print("   Recommended to follow 1→2→3→4 order for progressive learning")
        print()
        
        print("🚀 可用项目 | Available Projects:")
        print("-" * 50)
        
        for key, project in self.projects.items():
            print(f"{key}. {project['name']}")
            print(f"   📖 {project['description']}")
            print(f"   📖 {project['description_en']}")
            print()
        
        print("0. 退出程序 | Exit Program")
        print("A. 运行所有项目 | Run All Projects")
        print("H. 显示帮助信息 | Show Help Information")
        print("=" * 80)
    
    def run_script(self, script_path: str) -> bool:
        """
        运行指定脚本
        Run specified script
        
        Args:
            script_path: 脚本路径 | Script path
            
        Returns:
            是否成功运行 | Whether successfully run
        """
        full_path = os.path.join(self.project_dir, script_path)
        
        if not os.path.exists(full_path):
            print(f"❌ 文件不存在: {full_path}")
            print(f"❌ File does not exist: {full_path}")
            return False
        
        print(f"🏃 正在运行: {script_path}")
        print(f"🏃 Running: {script_path}")
        print("-" * 60)
        
        try:
            # 获取脚本目录
            script_dir = os.path.dirname(full_path)
            
            # 运行脚本
            result = subprocess.run([sys.executable, full_path], 
                                  cwd=script_dir,
                                  capture_output=False,
                                  text=True)
            
            if result.returncode == 0:
                print(f"✅ {script_path} 运行成功！")
                print(f"✅ {script_path} completed successfully!")
            else:
                print(f"❌ {script_path} 运行失败，返回码: {result.returncode}")
                print(f"❌ {script_path} failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ 运行 {script_path} 时发生错误: {str(e)}")
            print(f"❌ Error running {script_path}: {str(e)}")
            return False
        
        print("-" * 60)
        return True
    
    def run_project(self, project_key: str):
        """
        运行指定项目
        Run specified project
        
        Args:
            project_key: 项目键值 | Project key
        """
        if project_key not in self.projects:
            print(f"❌ 无效的项目选择: {project_key}")
            print(f"❌ Invalid project selection: {project_key}")
            return
        
        project = self.projects[project_key]
        
        print(f"\n🎯 开始运行项目: {project['name']}")
        print(f"🎯 Starting project: {project['name']}")
        print(f"📝 描述: {project['description']}")
        print(f"📝 Description: {project['description_en']}")
        print("=" * 80)
        
        success_count = 0
        total_count = len(project['scripts'])
        
        for i, script in enumerate(project['scripts'], 1):
            print(f"\n📁 步骤 {i}/{total_count}: 运行 {script}")
            print(f"📁 Step {i}/{total_count}: Running {script}")
            
            if self.run_script(script):
                success_count += 1
                print(f"✅ 步骤 {i} 完成")
                print(f"✅ Step {i} completed")
            else:
                print(f"❌ 步骤 {i} 失败")
                print(f"❌ Step {i} failed")
                
                user_input = input("是否继续下一步？(y/n) | Continue to next step? (y/n): ").lower()
                if user_input != 'y':
                    break
            
            # 添加一些间隔时间
            time.sleep(1)
        
        print(f"\n🎉 项目 '{project['name']}' 运行完成！")
        print(f"🎉 Project '{project['name']}' completed!")
        print(f"📊 成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        print(f"📊 Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    def run_all_projects(self):
        """
        运行所有项目
        Run all projects
        """
        print("\n🚀 开始运行所有项目")
        print("🚀 Starting to run all projects")
        print("=" * 80)
        
        total_projects = len(self.projects)
        completed_projects = 0
        
        for key in sorted(self.projects.keys()):
            print(f"\n📦 项目 {key}/{total_projects}")
            print(f"📦 Project {key}/{total_projects}")
            
            try:
                self.run_project(key)
                completed_projects += 1
                
                if key != max(self.projects.keys()):  # 不是最后一个项目
                    user_input = input("\n⏸️ 是否继续下一个项目？(y/n) | Continue to next project? (y/n): ").lower()
                    if user_input != 'y':
                        break
                        
            except KeyboardInterrupt:
                print("\n⏹️ 用户中断了程序执行")
                print("⏹️ User interrupted program execution")
                break
            except Exception as e:
                print(f"\n❌ 运行项目时发生错误: {str(e)}")
                print(f"❌ Error running project: {str(e)}")
        
        print(f"\n🎊 所有项目运行完成！完成率: {completed_projects}/{total_projects}")
        print(f"🎊 All projects completed! Completion rate: {completed_projects}/{total_projects}")
    
    def show_help(self):
        """
        显示帮助信息
        Show help information
        """
        print("\n📖 帮助信息 | Help Information")
        print("=" * 60)
        print()
        
        print("🎯 项目目标 | Project Objectives:")
        print("  通过实践项目深入理解LSTM和GRU的工作原理和应用场景")
        print("  Understand LSTM and GRU principles and applications through practical projects")
        print()
        
        print("📋 学习建议 | Learning Recommendations:")
        print("  1. 先阅读理论文档: ../05_LSTM_GRU/长短期记忆网络与门控循环单元.md")
        print("     First read theory: ../05_LSTM_GRU/长短期记忆网络与门控循环单元.md")
        print("  2. 按照项目编号顺序学习，循序渐进")
        print("     Follow project numbers in order for progressive learning")
        print("  3. 理解每个项目的代码注释和输出结果")
        print("     Understand code comments and output results of each project")
        print("  4. 尝试修改参数，观察不同配置的效果")
        print("     Try modifying parameters to observe effects of different configurations")
        print()
        
        print("⚠️ 注意事项 | Important Notes:")
        print("  • 确保已安装所有依赖包: pip install -r requirements.txt")
        print("    Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  • 某些项目可能需要较长的训练时间")
        print("    Some projects may require longer training time")
        print("  • 建议在有GPU的环境下运行以获得更好的性能")
        print("    Recommended to run in GPU environment for better performance")
        print("  • 所有生成的图表会保存在对应项目目录下")
        print("    All generated charts will be saved in corresponding project directories")
        print()
        
        print("🆘 遇到问题？| Having Issues?")
        print("  1. 检查Python版本（建议3.8+）")
        print("     Check Python version (recommended 3.8+)")
        print("  2. 检查PyTorch安装是否正确")
        print("     Check if PyTorch is installed correctly")
        print("  3. 确保有足够的内存空间")
        print("     Ensure sufficient memory space")
        print("  4. 查看具体错误信息，通常包含解决方案提示")
        print("     Check specific error messages, usually contain solution hints")
    
    def run(self):
        """
        运行主程序
        Run main program
        """
        while True:
            try:
                self.display_menu()
                
                choice = input("请选择项目 (输入数字或字母) | Please select project (enter number or letter): ").strip().upper()
                
                if choice == "0":
                    print("\n👋 谢谢使用！祝你学习愉快！")
                    print("👋 Thank you for using! Happy learning!")
                    break
                elif choice == "A":
                    self.run_all_projects()
                elif choice == "H":
                    self.show_help()
                elif choice in self.projects:
                    self.run_project(choice)
                else:
                    print("\n❌ 无效选择，请重新输入")
                    print("❌ Invalid selection, please try again")
                
                if choice not in ["0", "H"]:
                    input("\n按回车键继续... | Press Enter to continue...")
                    print("\n" * 2)  # 清屏效果
                
            except KeyboardInterrupt:
                print("\n\n👋 程序已退出，感谢使用！")
                print("👋 Program exited, thanks for using!")
                break
            except Exception as e:
                print(f"\n❌ 发生未知错误: {str(e)}")
                print(f"❌ Unknown error occurred: {str(e)}")
                input("按回车键继续... | Press Enter to continue...")


def check_environment():
    """
    检查运行环境
    Check runtime environment
    """
    print("🔍 检查运行环境 | Checking Runtime Environment")
    print("-" * 50)
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本 | Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️ 建议使用Python 3.8或更高版本")
        print("⚠️ Recommended to use Python 3.8 or higher")
    else:
        print("✅ Python版本符合要求")
        print("✅ Python version meets requirements")
    
    # 检查关键包
    required_packages = ['torch', 'numpy', 'pandas', 'matplotlib', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
            print(f"❌ {package} not installed")
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下包: {', '.join(missing_packages)}")
        print(f"⚠️ Missing packages: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        print("Please run: pip install -r requirements.txt")
        return False
    
    print("\n✅ 环境检查通过！")
    print("✅ Environment check passed!")
    return True


if __name__ == "__main__":
    print("🎓 LSTM/GRU深度学习实践项目启动器")
    print("🎓 LSTM/GRU Deep Learning Practice Project Launcher")
    print("=" * 80)
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请安装缺失的依赖包")
        print("❌ Environment check failed, please install missing dependencies")
        sys.exit(1)
    
    print("\n🚀 启动项目选择器...")
    print("🚀 Starting project selector...")
    time.sleep(1)
    
    # 运行主程序
    runner = ProjectRunner()
    runner.run() 