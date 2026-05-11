import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# 过滤不必要的警告
warnings.filterwarnings('ignore')

def analyze_and_plot_dataset_safe(file_path):
    # ---------------------------------------------------------
    # 1. dataload与预process
    # ---------------------------------------------------------
    try:
        # readdata
        df = pd.read_csv(file_path)
        
        # 简单重命名列(根据您的data结构)
        df.columns = ['SMILES', 'Ghex', 'Source']
        
        # 映射标签名称
        df['Source_Label'] = df['Source'].map({
            'E': 'Experimental (Exp)', 
            'A': 'ALOGPS (Calc)'
        })
        
        # --- 关bond修复: Extract为 NumPy 数组以避免版本冲突 ---
        ghex_values = df['Ghex'].values
        source_values = df['Source_Label'].values
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # ---------------------------------------------------------
    # 2. 统计描述 (打印output)
    # ---------------------------------------------------------
    print("--- Dataset Statistical Summary for Publication ---")
    summary = df.groupby('Source_Label')['Ghex'].describe().round(2)
    print(summary[['count', 'mean', 'std', 'min', 'max']])
    
    # ---------------------------------------------------------
    # 3. 绘图 (修复版)
    # ---------------------------------------------------------
    # 设置学术风格
    sns.set_theme(style="ticks", context="paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # 定义颜色
    colors = ["#2b8cbe", "#e6550d"] # 蓝、橙
    
    # [左图] 分布图 (Using .values 传入data)
    # note: 传入数组后, seaborn无法自动获取轴标签, 需手动设置
    sns.histplot(
        x=ghex_values, 
        hue=source_values,
        kde=True, 
        element="step", 
        stat="density",
        common_norm=False, 
        palette=colors, 
        alpha=0.25,
        ax=axes[0], 
        line_kws={'linewidth': 2}
    )
    axes[0].set_title('Distribution of Transfer Free Energy', fontweight='bold', pad=15)
    axes[0].set_xlabel('$\Delta G_{hex}$ (kcal/mol)')
    axes[0].set_ylabel('Density')
    # 修复图例标题(传入数组时有时会丢失)
    axes[0].legend(title='Data Source', labels=['ALOGPS (Calculated)', 'Experimental'])
    
    # [右图] 箱线图 (Using .values 传入data)
    sns.boxplot(
        x=source_values, 
        y=ghex_values,
        palette=colors, 
        width=0.4, 
        ax=axes[1],
        boxprops={'alpha': 0.8}, 
        saturation=0.9
    )
    axes[1].set_xlabel('')
    axes[1].set_ylabel('$\Delta G_{hex}$ (kcal/mol)')
    axes[1].set_title('Statistical Variability', fontweight='bold', pad=15)
    
    # 美化边框
    sns.despine()
    plt.tight_layout()
    
    # save图片
    output_filename = 'dataset_overview_fixed.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n[Success] Plot saved as '{output_filename}'")
    
    # for example果在非交互式环境(for example脚本)中运行, 取消下面这行的注释
    #plt.show()

# 运行代码
if __name__ == "__main__":
    # 请确保path正确, noteWindowspath中的反斜杠
    file_path = r'ML1_Ghex.csv'  # 或者Using您的绝对path 'E:\AutoCG\software\ML1_Ghex.csv'
    analyze_and_plot_dataset_safe(file_path)
