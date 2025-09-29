import os
import argparse

def list_files(startpath, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        # 使用栈而不是os.walk来更好地控制遍历顺序
        stack = [(startpath, 0)]
        
        while stack:
            path, level = stack.pop()
            
            # 获取所有条目并排序
            try:
                entries = os.listdir(path)
            except PermissionError:
                entries = []
            
            # 分离目录和文件，并分别排序
            dirs = []
            files = []
            for entry in entries:
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    dirs.append(entry)
                else:
                    files.append(entry)
            
            # 排序
            dirs.sort()
            files.sort()
            
            # 逆序添加到栈中，以便按正确顺序处理
            for dir_name in reversed(dirs):
                stack.append((os.path.join(path, dir_name), level + 1))
            
            # 计算缩进
            indent = '│   ' * level
            
            # 如果是根目录，特殊处理
            if path == startpath:
                f.write(f"{os.path.basename(path)}/\n")
            else:
                f.write(f"{indent}├── {os.path.basename(path)}/\n")
            
            # 写入文件
            subindent = '│   ' * (level + 1)
            for file in files:
                f.write(f"{subindent}├── {file}\n")

def main():
    parser = argparse.ArgumentParser(description='递归列出文件夹结构并保存到文本文件')
    parser.add_argument('folder_path', help='要遍历的文件夹路径')
    parser.add_argument('-o', '--output', default='folder_structure.txt', 
                        help='输出文件名(默认为 folder_structure.txt)')
    
    args = parser.parse_args()
    
    # 确保输出文件是txt格式
    if not args.output.endswith('.txt'):
        args.output += '.txt'
    
    # 检查文件夹是否存在
    if not os.path.isdir(args.folder_path):
        print(f"错误: 文件夹 '{args.folder_path}' 不存在")
        return
    
    list_files(args.folder_path, args.output)
    print(f"文件夹结构已保存到 {args.output}")

if __name__ == "__main__":
    main()