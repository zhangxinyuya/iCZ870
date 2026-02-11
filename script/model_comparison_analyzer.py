import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import sys
import os

# ==================== 配置参数 ====================
# 如果不使用命令行参数，可以直接在这里指定输入文件路径
INPUT_FILE = './screening_results_3models.xlsx'  # 默认输入文件名

# 输出配置
OUTPUT_DIR = './Model_Performance_Comparison_3Models'  # 输出目录
OUTPUT_FILENAME = 'Model_Performance_Comparison_3Models'  # 输出文件名（不含扩展名）
DPI = 600  # 图片分辨率

# 图表样式配置
FIGURE_SIZE = (20, 10)  # 图表尺寸（英寸）- 增加宽度以容纳3个模型
FONT_SIZE = 10  # 基础字体大小
TITLE_SIZE = 14  # 标题字体大小
LABEL_SIZE = 11  # 标签字体大小

# 颜色配置（3个模型）
COLOR_MODEL1 = '#4C72B0'  # iCZ870的颜色（蓝色）
COLOR_MODEL2 = '#DD8452'  # iCW773R的颜色（橙色）
COLOR_MODEL3 = '#55A868'  # 第3个模型的颜色（绿色）

CMAP_MODEL1 = 'Blues'  # iCZ870的热图颜色
CMAP_MODEL2 = 'Oranges'  # iCW773R的热图颜色
CMAP_MODEL3 = 'Greens'  # 第3个模型的热图颜色

# 模型名称（根据实际情况修改）
MODEL1_NAME = 'iCZ870'
MODEL2_NAME = 'iCW773R'
MODEL3_NAME = 'Model3'  # 👈 修改为你的第3个模型名称


# ==================== 函数定义 ====================

def setup_plot_style():
    """设置matplotlib绘图风格"""
    plt.rcParams['font.size'] = FONT_SIZE
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.labelsize'] = LABEL_SIZE
    plt.rcParams['axes.titlesize'] = TITLE_SIZE
    plt.rcParams['xtick.labelsize'] = FONT_SIZE - 1
    plt.rcParams['ytick.labelsize'] = FONT_SIZE - 1
    plt.rcParams['legend.fontsize'] = FONT_SIZE


def validate_input_file(filepath):
    """验证输入文件是否存在且格式正确"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"错误：找不到文件 '{filepath}'")

    if not filepath.endswith('.xlsx'):
        raise ValueError("错误：输入文件必须是 .xlsx 格式")

    print(f"✓ 找到输入文件: {filepath}")
    return True


def load_data(filepath):
    """
    从Excel文件加载数据

    参数:
        filepath: Excel文件路径

    返回:
        carbon_df: 碳源数据DataFrame
        nitrogen_df: 氮源数据DataFrame
    """
    try:
        # 读取Excel文件
        xl_file = pd.ExcelFile(filepath)

        # 检查工作表数量
        if len(xl_file.sheet_names) < 2:
            raise ValueError("错误：Excel文件必须包含至少2个工作表（碳源和氮源）")

        print(f"✓ 找到 {len(xl_file.sheet_names)} 个工作表: {xl_file.sheet_names}")

        # 读取Sheet 1（碳源）和Sheet 2（氮源）
        carbon_df = pd.read_excel(filepath, sheet_name=0)  # 第一个sheet
        nitrogen_df = pd.read_excel(filepath, sheet_name=1)  # 第二个sheet

        print(f"✓ 碳源数据: {carbon_df.shape[0]} 行, {carbon_df.shape[1]} 列")
        print(f"✓ 氮源数据: {nitrogen_df.shape[0]} 行, {nitrogen_df.shape[1]} 列")

        # 自动检测列名
        print("\n检测到的列名:")
        print(f"  碳源: {list(carbon_df.columns)}")
        print(f"  氮源: {list(nitrogen_df.columns)}")

        # 验证必需的列是否存在
        required_columns = ['Growth']

        # 查找包含 "Sim in" 的列名（模型预测列）
        sim_columns_carbon = [col for col in carbon_df.columns if 'Sim in' in col]
        sim_columns_nitrogen = [col for col in nitrogen_df.columns if 'Sim in' in col]

        if len(sim_columns_carbon) < 3:
            raise ValueError(f"错误：碳源数据至少需要3个'Sim in'列，当前只有{len(sim_columns_carbon)}个")

        if len(sim_columns_nitrogen) < 3:
            raise ValueError(f"错误：氮源数据至少需要3个'Sim in'列，当前只有{len(sim_columns_nitrogen)}个")

        print(f"\n✓ 找到3个模型的预测列:")
        print(f"  碳源: {sim_columns_carbon[:3]}")
        print(f"  氮源: {sim_columns_nitrogen[:3]}")

        return carbon_df, nitrogen_df, sim_columns_carbon[:3], sim_columns_nitrogen[:3]

    except Exception as e:
        print(f"读取数据时出错: {e}")
        raise


def process_data(df, substrate_name, sim_col1, sim_col2, sim_col3):
    """
    处理数据并计算混淆矩阵和准确率（3个模型）

    参数:
        df: 输入的DataFrame
        substrate_name: 底物类型名称（用于输出信息）
        sim_col1, sim_col2, sim_col3: 3个模型的预测列名

    返回:
        cm1, cm2, cm3: 3个模型的混淆矩阵
        acc1, acc2, acc3: 3个模型的准确率
    """
    print(f"\n处理{substrate_name}数据...")

    # 复制数据以避免修改原始数据
    df_clean = df.copy()

    # 移除任何模型预测值为NaN的行
    original_count = len(df_clean)
    df_clean = df_clean[
        df_clean[sim_col1].notna() &
        df_clean[sim_col2].notna() &
        df_clean[sim_col3].notna()
        ]
    removed_count = original_count - len(df_clean)

    if removed_count > 0:
        print(f"  - 移除了 {removed_count} 行缺失数据")

    print(f"  - 有效数据: {len(df_clean)} 行")

    # 转换为布尔类型
    y_true = df_clean['Growth'].apply(lambda x: str(x).strip().upper() == 'TRUE').values
    y_pred1 = df_clean[sim_col1].apply(lambda x: str(x).strip().upper() == 'TRUE').values
    y_pred2 = df_clean[sim_col2].apply(lambda x: str(x).strip().upper() == 'TRUE').values
    y_pred3 = df_clean[sim_col3].apply(lambda x: str(x).strip().upper() == 'TRUE').values

    # 计算混淆矩阵
    cm1 = confusion_matrix(y_true, y_pred1)
    cm2 = confusion_matrix(y_true, y_pred2)
    cm3 = confusion_matrix(y_true, y_pred3)

    # 计算准确率
    acc1 = accuracy_score(y_true, y_pred1)
    acc2 = accuracy_score(y_true, y_pred2)
    acc3 = accuracy_score(y_true, y_pred3)

    # 输出统计信息
    tn1, fp1, fn1, tp1 = cm1.ravel()
    tn2, fp2, fn2, tp2 = cm2.ravel()
    tn3, fp3, fn3, tp3 = cm3.ravel()

    print(f"  模型1 - TP:{tp1}, TN:{tn1}, FP:{fp1}, FN:{fn1}, Accuracy:{acc1:.2%}")
    print(f"  模型2 - TP:{tp2}, TN:{tn2}, FP:{fp2}, FN:{fn2}, Accuracy:{acc2:.2%}")
    print(f"  模型3 - TP:{tp3}, TN:{tn3}, FP:{fp3}, FN:{fn3}, Accuracy:{acc3:.2%}")

    return cm1, cm2, cm3, acc1, acc2, acc3


def create_comparison_figure(cm_c1, cm_c2, cm_c3, acc_c1, acc_c2, acc_c3,
                             cm_n1, cm_n2, cm_n3, acc_n1, acc_n2, acc_n3,
                             model_names, output_path):
    """
    创建完整的对比图表（3模型版本）

    参数:
        cm_c1, cm_c2, cm_c3: 碳源的3个混淆矩阵
        acc_c1, acc_c2, acc_c3: 碳源的3个准确率
        cm_n1, cm_n2, cm_n3: 氮源的3个混淆矩阵
        acc_n1, acc_n2, acc_n3: 氮源的3个准确率
        model_names: 模型名称列表
        output_path: 输出路径前缀（不含扩展名）
    """
    print("\n生成3模型对比图表...")

    # 创建图表 - 2行4列布局
    fig = plt.figure(figsize=FIGURE_SIZE)
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25,
                          left=0.05, right=0.98, top=0.92, bottom=0.08)

    colors = [COLOR_MODEL1, COLOR_MODEL2, COLOR_MODEL3]
    cmaps = [CMAP_MODEL1, CMAP_MODEL2, CMAP_MODEL3]

    # ============ 第一行：碳源 ============

    # 碳源 - 模型1混淆矩阵
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm_c1, annot=True, fmt='d', cmap=cmaps[0], cbar=False,
                square=True, linewidths=2, linecolor='white',
                xticklabels=['No', 'Growth'],
                yticklabels=['No', 'Growth'], ax=ax1,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax1.set_xlabel('Predicted', fontweight='bold')
    ax1.set_ylabel('Experimental', fontweight='bold')
    ax1.set_title(f'Carbon - {model_names[0]}', fontweight='bold', fontsize=12)
    ax1.text(0.5, -0.2, f'Acc: {acc_c1:.2%}', transform=ax1.transAxes,
             fontsize=10, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # 碳源 - 模型2混淆矩阵
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(cm_c2, annot=True, fmt='d', cmap=cmaps[1], cbar=False,
                square=True, linewidths=2, linecolor='white',
                xticklabels=['No', 'Growth'],
                yticklabels=['No', 'Growth'], ax=ax2,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax2.set_xlabel('Predicted', fontweight='bold')
    ax2.set_ylabel('Experimental', fontweight='bold')
    ax2.set_title(f'Carbon - {model_names[1]}', fontweight='bold', fontsize=12)
    ax2.text(0.5, -0.2, f'Acc: {acc_c2:.2%}', transform=ax2.transAxes,
             fontsize=10, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='#FFE5CC', alpha=0.7))

    # 碳源 - 模型3混淆矩阵
    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(cm_c3, annot=True, fmt='d', cmap=cmaps[2], cbar=False,
                square=True, linewidths=2, linecolor='white',
                xticklabels=['No', 'Growth'],
                yticklabels=['No', 'Growth'], ax=ax3,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax3.set_xlabel('Predicted', fontweight='bold')
    ax3.set_ylabel('Experimental', fontweight='bold')
    ax3.set_title(f'Carbon - {model_names[2]}', fontweight='bold', fontsize=12)
    ax3.text(0.5, -0.2, f'Acc: {acc_c3:.2%}', transform=ax3.transAxes,
             fontsize=10, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # 碳源 - Accuracy对比柱状图
    ax4 = fig.add_subplot(gs[0, 3])
    accuracies_c = [acc_c1, acc_c2, acc_c3]
    x_pos = np.arange(len(model_names))
    bars = ax4.bar(x_pos, accuracies_c, color=colors, edgecolor='black', linewidth=2)

    for bar, acc in zip(bars, accuracies_c):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{acc:.2%}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(model_names, rotation=15, ha='right')
    ax4.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax4.set_title('Carbon Source\nAccuracy Comparison', fontweight='bold', fontsize=12)
    ax4.set_ylim([0, 1.08])
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # ============ 第二行：氮源 ============

    # 氮源 - 模型1混淆矩阵
    ax5 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cm_n1, annot=True, fmt='d', cmap=cmaps[0], cbar=False,
                square=True, linewidths=2, linecolor='white',
                xticklabels=['No', 'Growth'],
                yticklabels=['No', 'Growth'], ax=ax5,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax5.set_xlabel('Predicted', fontweight='bold')
    ax5.set_ylabel('Experimental', fontweight='bold')
    ax5.set_title(f'Nitrogen - {model_names[0]}', fontweight='bold', fontsize=12)
    ax5.text(0.5, -0.2, f'Acc: {acc_n1:.2%}', transform=ax5.transAxes,
             fontsize=10, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # 氮源 - 模型2混淆矩阵
    ax6 = fig.add_subplot(gs[1, 1])
    sns.heatmap(cm_n2, annot=True, fmt='d', cmap=cmaps[1], cbar=False,
                square=True, linewidths=2, linecolor='white',
                xticklabels=['No', 'Growth'],
                yticklabels=['No', 'Growth'], ax=ax6,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax6.set_xlabel('Predicted', fontweight='bold')
    ax6.set_ylabel('Experimental', fontweight='bold')
    ax6.set_title(f'Nitrogen - {model_names[1]}', fontweight='bold', fontsize=12)
    ax6.text(0.5, -0.2, f'Acc: {acc_n2:.2%}', transform=ax6.transAxes,
             fontsize=10, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='#FFE5CC', alpha=0.7))

    # 氮源 - 模型3混淆矩阵
    ax7 = fig.add_subplot(gs[1, 2])
    sns.heatmap(cm_n3, annot=True, fmt='d', cmap=cmaps[2], cbar=False,
                square=True, linewidths=2, linecolor='white',
                xticklabels=['No', 'Growth'],
                yticklabels=['No', 'Growth'], ax=ax7,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax7.set_xlabel('Predicted', fontweight='bold')
    ax7.set_ylabel('Experimental', fontweight='bold')
    ax7.set_title(f'Nitrogen - {model_names[2]}', fontweight='bold', fontsize=12)
    ax7.text(0.5, -0.2, f'Acc: {acc_n3:.2%}', transform=ax7.transAxes,
             fontsize=10, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # 氮源 - Accuracy对比柱状图
    ax8 = fig.add_subplot(gs[1, 3])
    accuracies_n = [acc_n1, acc_n2, acc_n3]
    bars = ax8.bar(x_pos, accuracies_n, color=colors, edgecolor='black', linewidth=2)

    for bar, acc in zip(bars, accuracies_n):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                 f'{acc:.2%}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(model_names, rotation=15, ha='right')
    ax8.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax8.set_title('Nitrogen Source\nAccuracy Comparison', fontweight='bold', fontsize=12)
    ax8.set_ylim([0, 1.08])
    ax8.grid(axis='y', alpha=0.3, linestyle='--')
    ax8.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # 保存图表
    png_path = f"{output_path}.png"
    pdf_path = f"{output_path}.pdf"

    fig.savefig(png_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')

    plt.close(fig)

    print(f"✓ 保存PNG: {png_path}")
    print(f"✓ 保存PDF: {pdf_path}")

    return png_path, pdf_path


def print_summary(acc_c1, acc_c2, acc_c3, acc_n1, acc_n2, acc_n3, model_names):
    """打印结果摘要（3个模型）"""
    print("\n" + "=" * 80)
    print(" " * 25 + "ACCURACY SUMMARY")
    print("=" * 80)

    print(f"\nCARBON SOURCE:")
    print(
        f"  {model_names[0]:<12}: {acc_c1:>7.2%}  {'⭐' * 5 if acc_c1 >= 0.9 else '⭐' * 4 if acc_c1 >= 0.8 else '⭐' * 3}")
    print(
        f"  {model_names[1]:<12}: {acc_c2:>7.2%}  {'⭐' * 5 if acc_c2 >= 0.9 else '⭐' * 4 if acc_c2 >= 0.8 else '⭐' * 3}")
    print(
        f"  {model_names[2]:<12}: {acc_c3:>7.2%}  {'⭐' * 5 if acc_c3 >= 0.9 else '⭐' * 4 if acc_c3 >= 0.8 else '⭐' * 3}")

    # 找出最佳模型
    best_idx_c = np.argmax([acc_c1, acc_c2, acc_c3])
    print(f"\n  🏆 Best Model: {model_names[best_idx_c]} ({[acc_c1, acc_c2, acc_c3][best_idx_c]:.2%})")

    print(f"\nNITROGEN SOURCE:")
    print(
        f"  {model_names[0]:<12}: {acc_n1:>7.2%}  {'⭐' * 5 if acc_n1 >= 0.9 else '⭐' * 4 if acc_n1 >= 0.8 else '⭐' * 3}")
    print(
        f"  {model_names[1]:<12}: {acc_n2:>7.2%}  {'⭐' * 5 if acc_n2 >= 0.9 else '⭐' * 4 if acc_n2 >= 0.8 else '⭐' * 3}")
    print(
        f"  {model_names[2]:<12}: {acc_n3:>7.2%}  {'⭐' * 5 if acc_n3 >= 0.9 else '⭐' * 4 if acc_n3 >= 0.8 else '⭐' * 3}")

    # 找出最佳模型
    best_idx_n = np.argmax([acc_n1, acc_n2, acc_n3])
    print(f"\n  🏆 Best Model: {model_names[best_idx_n]} ({[acc_n1, acc_n2, acc_n3][best_idx_n]:.2%})")

    # 平均性能
    avg_acc = [(acc_c1 + acc_n1) / 2, (acc_c2 + acc_n2) / 2, (acc_c3 + acc_n3) / 2]
    print(f"\nOVERALL AVERAGE:")
    for i, name in enumerate(model_names):
        print(f"  {name:<12}: {avg_acc[i]:>7.2%}")

    best_overall = np.argmax(avg_acc)
    print(f"\n  🏆 Overall Best: {model_names[best_overall]} ({avg_acc[best_overall]:.2%})")

    print("=" * 80)


def save_summary_to_file(acc_c1, acc_c2, acc_c3, acc_n1, acc_n2, acc_n3,
                         model_names, output_dir):
    """保存结果摘要到文本文件（3个模型）"""
    summary_path = os.path.join(output_dir, 'analysis_summary_3models.txt')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 25 + "ACCURACY SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("CARBON SOURCE:\n")
        f.write(f"  {model_names[0]:<12}: {acc_c1:.4f} ({acc_c1:.2%})\n")
        f.write(f"  {model_names[1]:<12}: {acc_c2:.4f} ({acc_c2:.2%})\n")
        f.write(f"  {model_names[2]:<12}: {acc_c3:.4f} ({acc_c3:.2%})\n")

        best_idx_c = np.argmax([acc_c1, acc_c2, acc_c3])
        f.write(f"\n  Best Model: {model_names[best_idx_c]}\n")

        f.write("\nNITROGEN SOURCE:\n")
        f.write(f"  {model_names[0]:<12}: {acc_n1:.4f} ({acc_n1:.2%})\n")
        f.write(f"  {model_names[1]:<12}: {acc_n2:.4f} ({acc_n2:.2%})\n")
        f.write(f"  {model_names[2]:<12}: {acc_n3:.4f} ({acc_n3:.2%})\n")

        best_idx_n = np.argmax([acc_n1, acc_n2, acc_n3])
        f.write(f"\n  Best Model: {model_names[best_idx_n]}\n")

        avg_acc = [(acc_c1 + acc_n1) / 2, (acc_c2 + acc_n2) / 2, (acc_c3 + acc_n3) / 2]
        f.write("\nOVERALL AVERAGE:\n")
        for i, name in enumerate(model_names):
            f.write(f"  {name:<12}: {avg_acc[i]:.4f} ({avg_acc[i]:.2%})\n")

        best_overall = np.argmax(avg_acc)
        f.write(f"\n  Overall Best: {model_names[best_overall]}\n")

        f.write("=" * 80 + "\n")

    print(f"✓ 保存摘要: {summary_path}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" " * 18 + "三模型代谢预测性能对比分析")
    print("=" * 80 + "\n")

    # 1. 确定输入文件
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = INPUT_FILE

    print(f"输入文件: {input_file}")

    try:
        # 2. 验证输入文件
        validate_input_file(input_file)

        # 3. 设置绘图风格
        setup_plot_style()

        # 4. 加载数据
        carbon_df, nitrogen_df, sim_cols_c, sim_cols_n = load_data(input_file)

        # 提取模型名称
        model_names = [col.replace('Sim in ', '') for col in sim_cols_c]
        print(f"\n✓ 检测到的模型名称: {model_names}")

        # 5. 处理数据
        cm_c1, cm_c2, cm_c3, acc_c1, acc_c2, acc_c3 = process_data(
            carbon_df, "碳源", sim_cols_c[0], sim_cols_c[1], sim_cols_c[2]
        )
        cm_n1, cm_n2, cm_n3, acc_n1, acc_n2, acc_n3 = process_data(
            nitrogen_df, "氮源", sim_cols_n[0], sim_cols_n[1], sim_cols_n[2]
        )

        # 6. 创建输出目录
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"\n✓ 输出目录: {OUTPUT_DIR}")

        # 7. 生成图表
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        create_comparison_figure(
            cm_c1, cm_c2, cm_c3, acc_c1, acc_c2, acc_c3,
            cm_n1, cm_n2, cm_n3, acc_n1, acc_n2, acc_n3,
            model_names, output_path
        )

        # 8. 打印和保存摘要
        print_summary(acc_c1, acc_c2, acc_c3, acc_n1, acc_n2, acc_n3, model_names)
        save_summary_to_file(acc_c1, acc_c2, acc_c3, acc_n1, acc_n2, acc_n3,
                             model_names, OUTPUT_DIR)

        print("\n" + "=" * 80)
        print("✅ 三模型分析完成！所有文件已保存到输出目录。")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()