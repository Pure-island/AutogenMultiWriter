import json
import re
import argparse
from pathlib import Path

def slugify(s: str) -> str:
    """将字符串转换为适合文件名的格式"""
    s = s.lower()
    s = re.sub(r"[^\w\s+-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s[:120]

def generate_chapter_readme(chapter_dir: Path, chapter_title: str, sections: list):
    """生成章节的README.md文件"""
    readme_content = f"# {chapter_title}\n\n"
    readme_content += f"本章节包含以下内容：\n\n"
    for section_idx, section in enumerate(sections, 1):
        readme_content += f"{section_idx}. {section['title']} - {section.get('desc', '')}\n"
    
    readme_content += "\n请从左侧目录选择具体小节开始学习。\n"
    
    try:
        with open(chapter_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        return True
    except Exception as e:
        print(f"错误: 写入章节README.md时发生错误 - {e}")
        return False

def generate_gitbook_files(json_path: Path):
    """根据JSON目录文件生成GitBook文件"""
    # 检查文件是否存在
    if not json_path.exists():
        print(f"错误: JSON文件 '{json_path}' 不存在")
        return False
    
    # 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            catalog = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: JSON文件格式不正确 - {e}")
        return False
    except Exception as e:
        print(f"错误: 读取文件时发生错误 - {e}")
        return False
    
    # 验证JSON结构
    if 'title' not in catalog or 'chapters' not in catalog:
        print("错误: JSON文件缺少必要的字段 ('title' 或 'chapters')")
        return False
    
    # 获取JSON文件所在目录
    base_dir = json_path.parent
    
    # 生成主README.md
    readme_content = f"# {catalog['title']}\n\n"
    readme_content += "本书详细介绍了Unity游戏开发中的网络技术，涵盖从基础概念到高级实现的全面内容。\n\n"
    readme_content += "## 目录结构\n\n"
    
    for chapter_idx, chapter in enumerate(catalog['chapters'], 1):
        readme_content += f"{chapter_idx}. {chapter['title']}\n"
    
    try:
        with open(base_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
    except Exception as e:
        print(f"错误: 写入主README.md时发生错误 - {e}")
        return False
    
    # 生成SUMMARY.md
    summary_content = "# Summary\n\n"
    summary_content += "* [简介](README.md)\n\n"
    
    # 创建章节文件夹并生成章节README.md
    for chapter_idx, chapter in enumerate(catalog['chapters'], 1):
        chapter_title = chapter['title']
        chapter_dir_name = f"{chapter_idx:02d}_{slugify(chapter_title)}"
        chapter_dir = base_dir / chapter_dir_name
        
        # 创建章节文件夹
        try:
            chapter_dir.mkdir(exist_ok=True)
        except Exception as e:
            print(f"错误: 创建章节文件夹 '{chapter_dir}' 时发生错误 - {e}")
            return False
        
        # 生成章节README.md
        if not generate_chapter_readme(chapter_dir, chapter_title, chapter['sections']):
            return False
        
        # 添加章节条目到SUMMARY.md
        summary_content += f"* [{chapter_title}]({chapter_dir_name}/README.md)\n"
        
        # 添加小节条目到SUMMARY.md
        for section_idx, section in enumerate(chapter['sections'], 1):
            section_title = section['title']
            filename = f"{chapter_idx:02d}_{section_idx:02d}_{slugify(chapter_title)}_{slugify(section_title)}.md"
            summary_content += f"  * [{section_title}]({chapter_dir_name}/{filename})\n"
        
        summary_content += "\n"
    
    # 写入SUMMARY.md
    try:
        with open(base_dir / "SUMMARY.md", 'w', encoding='utf-8') as f:
            f.write(summary_content)
    except Exception as e:
        print(f"错误: 写入SUMMARY.md时发生错误 - {e}")
        return False
    
    print("GitBook文件生成完成！")
    print(f"主README.md 已保存至: {base_dir / 'README.md'}")
    print(f"SUMMARY.md 已保存至: {base_dir / 'SUMMARY.md'}")
    
    # 统计生成的章节README.md数量
    chapter_count = len(catalog['chapters'])
    print(f"已生成 {chapter_count} 个章节的README.md文件")
    
    return True

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='根据JSON目录文件生成GitBook格式的README.md和SUMMARY.md')
    parser.add_argument('json_file', type=str, help='输入的JSON目录文件路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    
    if args.verbose:
        print(f"正在处理文件: {json_path}")
    
    success = generate_gitbook_files(json_path)
    
    if args.verbose:
        if success:
            print("处理完成")
        else:
            print("处理失败")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())