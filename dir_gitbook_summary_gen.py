import os
import subprocess
import argparse
from pathlib import Path

def find_toc_files(root_dir):
    """递归查找所有以00_toc开头的JSON文件"""
    toc_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith("00_toc") and file.endswith(".json"):
                toc_files.append(Path(root) / file)
    return toc_files

def process_toc_files(root_dir, verbose=False):
    """处理所有找到的TOC文件"""
    toc_files = find_toc_files(root_dir)
    
    if not toc_files:
        print(f"在目录 {root_dir} 及其子目录中未找到任何以00_toc开头的JSON文件")
        return 0
    
    print(f"找到 {len(toc_files)} 个TOC文件:")
    for toc_file in toc_files:
        print(f"  - {toc_file}")
    
    success_count = 0
    for toc_file in toc_files:
        if verbose:
            print(f"\n正在处理: {toc_file}")
        
        try:
            # 调用gitbook_summary_gen.py处理当前TOC文件
            result = subprocess.run(
                ["python", "gitbook_summary_gen.py", str(toc_file)],
                capture_output=True,
                text=True,
                check=True
            )
            
            if verbose:
                print(result.stdout)
                if result.stderr:
                    print(f"警告: {result.stderr}")
            
            success_count += 1
            if verbose:
                print(f"成功处理: {toc_file}")
                
        except subprocess.CalledProcessError as e:
            print(f"处理 {toc_file} 时出错: {e}")
            if verbose:
                print(f"错误输出: {e.stderr}")
        except Exception as e:
            print(f"处理 {toc_file} 时发生意外错误: {e}")
    
    print(f"\n处理完成: 成功 {success_count}/{len(toc_files)} 个文件")
    return success_count

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='递归遍历文件夹并处理所有以00_toc开头的JSON目录文件')
    parser.add_argument('root_dir', type=str, help='要遍历的根目录路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    
    if not root_dir.exists():
        print(f"错误: 目录 '{root_dir}' 不存在")
        return 1
    
    if not root_dir.is_dir():
        print(f"错误: '{root_dir}' 不是目录")
        return 1
    
    if args.verbose:
        print(f"开始遍历目录: {root_dir}")
    
    success_count = process_toc_files(root_dir, args.verbose)
    
    if args.verbose:
        print(f"处理完成，成功处理 {success_count} 个文件")
    
    return 0 if success_count > 0 else 1

if __name__ == "__main__":
    exit(main())