#!/bin/bash

# ================= 配置区域 =================
# 源文件目录
SRC_DIR="/data2/simlingo/targzData"
# 解压目标目录
DEST_DIR="/data2/simlingo/"
# 文件匹配模式
FILE_PATTERN="data*.tar.gz"
# ===========================================

# 1. 检查目录是否存在
if [ ! -d "$SRC_DIR" ]; then
    echo "错误: 源目录 $SRC_DIR 不存在！"
    exit 1
fi

# 创建目标目录
mkdir -p "$DEST_DIR"

echo "正在扫描文件..."

# 2. 将所有匹配的文件路径存入数组（处理文件名带空格的情况）
# 使用 find 查找并排序，存入 files 数组
mapfile -t files < <(find "$SRC_DIR" -name "$FILE_PATTERN" | sort)

# 3. 统计文件总数
total_files=${#files[@]}
current_index=0

if [ "$total_files" -eq 0 ]; then
    echo "未在 $SRC_DIR 中找到匹配 $FILE_PATTERN 的文件。"
    exit 0
fi

echo "共发现 $total_files 个压缩包，准备开始解压..."
echo "输出目录: $DEST_DIR"
echo "" # 空一行

# 获取开始时间
start_time=$(date +%s)

# 4. 开始循环解压
for file in "${files[@]}"; do
    ((current_index++))

    # 计算百分比
    percent=$((current_index * 100 / total_files))
    
    # 计算进度条长度 (总长50个字符)
    bar_len=$((percent / 2))
    
    # 生成进度条字符串
    # 使用 printf 生成 # 号
    bar=$(printf "%-${bar_len}s" "#" | tr ' ' '#')
    # 补充剩余空格
    empty_len=$((50 - bar_len))
    
    # 获取文件名（用于显示）
    filename=$(basename "$file")

    # 5. 打印进度条
    # \r 让光标回到行首，实现原地刷新
    # \033[K 清除当前行光标后的内容，防止残留字符
    printf "\r[%-50s] %d%% (%d/%d) 正在解压: %s\033[K" "$bar" "$percent" "$current_index" "$total_files" "$filename"

    # 6. 执行解压 (核心命令)
    # 这里的 -C 指定解压目录
    tar -xzf "$file" -C "$DEST_DIR"

done

# 结束换行
echo ""
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "----------------------------------------"
echo "✅ 全部完成！"
echo "耗时: ${duration} 秒"
echo "文件已解压至: $DEST_DIR"