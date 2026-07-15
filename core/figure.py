

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.offline import plot
from plotly.subplots import make_subplots

from core.market_essentials import import_index_data, get_most_stock_by_year
from core.model.backtest_config import BacktestConfig
from core.utils.color_kit import ColorKit
from core.utils.log_kit import logger
from core.utils.misc_kit import save_csv_safely


def draw_equity_curve_plotly(
    df,
    data_dict,
    date_col=None,
    right_axis=None,
    pic_size=None,
    chg=False,
    title=None,
    rtn_add=pd.DataFrame(),
    desc=None,
    to_zero=True,
    show_subplots=False,
):
    if pic_size is None:
        pic_size = [1500, 626]

    draw_df = df.copy()

    # 设置时间序列
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index

    # 绘制左轴数据
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,  # 共享 x 轴，主，子图共同变化
        vertical_spacing=0.02,  # 减少主图和子图之间的间距
        row_heights=[0.8, 0.2],  # 主图高度占 70%，子图各占 10%
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    color_i = 0
    for key in data_dict:
        col_name = data_dict[key]
        if chg:
            draw_df[col_name] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        else:
            draw_df[col_name] = draw_df[data_dict[key]]
        line_color = ColorKit.cycle_color_list[color_i % len(ColorKit.cycle_color_list)]
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[col_name], name=key, line=dict(color=line_color)), row=1, col=1)
        color_i += 1

    # 绘制右轴数据
    if right_axis:
        key = list(right_axis.keys())[0]
        if to_zero:
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=draw_df[right_axis[key]],
                    name=key + "(右轴)",
                    opacity=0.1,
                    marker=dict(color=ColorKit.GOLD, opacity=1),
                    line=dict(width=0),
                    fill="tozeroy",
                    yaxis="y2",
                )
            )  # 标明设置一个不同于trace1的一个坐标轴
        else:
            fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + "(右轴)", yaxis="y2"))

    # 绘制子图
    if show_subplots:
        # 子图：按照 matplotlib stackplot 风格实现堆叠图
        # 仓位占比
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=draw_df["实际杠杆"],
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor=ColorKit.LIGHT_BLUE,
                name="仓位占比",
                hovertemplate="仓位占比: %{customdata:.4f}<extra></extra>",
                customdata=draw_df["实际杠杆"],  # 使用原始比例值
            ),
            row=2,
            col=1,
        )

        # 空仓占比
        fig.add_trace(
            go.Scatter(
                x=time_data,
                # 必须要这样才能实现堆叠的效果，不然会出现坑坑洼洼的
                y=1 - draw_df["实际杠杆"] + draw_df["实际杠杆"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=ColorKit.LIGHT_GRAY,
                name="空仓占比",
                hovertemplate="空仓占比: %{customdata:.4f}<extra></extra>",
                customdata=1 - draw_df["实际杠杆"],  # 使用原始比例值
            ),
            row=2,
            col=1,
        )

        # 调整子图的宽度，不然会超出主图的宽度
        fig.update_xaxes(domain=[0, 0.73], row=2, col=1)

    fig.update_layout(
        template="none",
        width=pic_size[0],
        height=pic_size[1],
        title={"text": title, "x": 170 / pic_size[0], "xanchor": "left"},
        hovermode="x unified",
        hoverlabel=dict(bgcolor=ColorKit.WHITE),
        margin=dict(t=50),
        annotations=[
            dict(
                text=desc,
                xref="paper",
                yref="paper",
                x=0.005,
                y=1.05,
                showarrow=False,
                font=dict(size=12, color="black"),
                align="left",
                bgcolor=ColorKit.WHITE,
            )
        ],
        paper_bgcolor=ColorKit.WHITE,  # 整个图表区域背景色
        plot_bgcolor=ColorKit.WHITE,  # 主图背景色
        legend=dict(
            x=0.8,
            y=1,
            itemsizing="constant",
            traceorder="normal",
            bordercolor=ColorKit.SLATE_GRAY,
            borderwidth=2,
            itemwidth=32,
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            # 关键：设置legend的symbol为方块
            entrywidth=80,
            entrywidthmode="pixels",
            # symbol shape
            # 下面这行是关键
            # 但plotly官方没有直接legend_symbol_shape参数，需在trace里设置legendgroup和marker.symbol
        ),
    )
    # 统一设置所有trace的legend symbol为方块
    for trace in fig.data:
        if hasattr(trace, "marker"):
            if trace.marker is None:
                trace.marker = {}
            trace.marker["symbol"] = "square"

    x_limit = 0.98 if rtn_add.empty else 0.73
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="线性 y轴", method="relayout", args=[{"yaxis.type": "linear"}]),
                    dict(label="Log y轴", method="relayout", args=[{"yaxis.type": "log"}]),
                ]
            )
        ],
        xaxis=dict(domain=[0.0, x_limit]),
        paper_bgcolor=ColorKit.WHITE,  # 整个图表区域背景色
        plot_bgcolor=ColorKit.WHITE,  # 主图背景色
    )

    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="x domain",
                yref="y domain",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color=ColorKit.LIGHT_GRAY, width=1),
                layer="below",
            ),
        ]
    )

    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="solid", spikethickness=1)
    fig.update_xaxes(
        showspikes=True, spikemode="across+marker", spikesnap="cursor", spikedash="solid", spikethickness=1
    )
    if not rtn_add.empty:
        # 把rtn放进图里
        rtn_add = rtn_add.T
        rtn_add["最大回撤开始时间"] = rtn_add["最大回撤开始时间"].str.replace("00:00:00", "")
        rtn_add["最大回撤结束时间"] = rtn_add["最大回撤结束时间"].str.replace("00:00:00", "")
        rtn_add.columns = rtn_add.columns.str.replace("(", "（").str.replace(")", "）")
        rtn_add = rtn_add.T
        header_list = ["项目", "策略表现"] if rtn_add.shape[1] == 1 else ["项目"] + list(rtn_add.columns)
        rtn_add.reset_index(drop=False, inplace=True)
        # 交替设置表格行的背景色：一行为原色，一行为lavender
        num_rows = rtn_add.shape[0]
        cell_fill_colors = []
        for i in range(num_rows):
            if i % 2 == 0:
                cell_fill_colors.append(ColorKit.WHITE)
            else:
                cell_fill_colors.append(ColorKit.SHELL)
        table_trace = go.Table(
            header=dict(
                values=header_list,
                fill_color=ColorKit.SLATE_GRAY,
                font=dict(color=ColorKit.WHITE, weight="bold"),
                line=dict(width=1, color=ColorKit.SLATE_GRAY),
            ),
            cells=dict(
                values=rtn_add.T.values.tolist(),
                fill_color=[cell_fill_colors] * rtn_add.T.shape[0],
                line=dict(width=1, color=ColorKit.SLATE_GRAY),
            ),
            domain=dict(x=[0.77, 1.0], y=[0.0, 0.705]),
        )
        fig.add_trace(table_trace)
        # 图例调一下位置
        fig.update_layout(legend=dict(x=0.8, y=1))

    return_fig = plot(fig, include_plotlyjs=True, output_type="div")
    return return_fig


def draw_table(table_df, width=1500, row_height=26, title=None, max_line_length=200, first_col_width=53):
    """
    绘制表格
    :param table_df: 表格数据
    :param width: 图片的宽度
    :param row_height: 表格的默认行高
    :param title: 表格的标题
    :param max_line_length: 每行文字的最大长度（超过则换行）
    :param first_col_width: 第一列的固定宽度
    :return: 返回表格的 HTML div
    """
    # 计算每一列的最大宽度
    max_widths = [max([len(str(value)) for value in table_df[col]]) for col in table_df.columns]

    # 设置列宽
    # 第一列固定宽度，其他列按比例分配剩余宽度
    remaining_width = width - first_col_width  # 剩余宽度
    other_col_widths = [max_width / np.sum(max_widths[1:]) * remaining_width for max_width in max_widths[1:]]
    column_width = [first_col_width] + other_col_widths  # 第一列固定宽度，其他列按比例分配

    # 自动换行函数
    def wrap_text(text, max_length):
        if not isinstance(text, str):
            text = str(text)
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_length:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return "<br>".join(lines)

    # 对表格中的每个单元格应用自动换行
    wrapped_values = table_df.map(lambda x: wrap_text(x, max_line_length))

    # 计算每行的最大高度（根据换行后的行数）
    row_heights = []
    for _, row in wrapped_values.iterrows():
        max_lines = max([len(str(cell).split("<br>")) for cell in row])
        row_heights.append(max_lines * row_height)

    # 计算表格的总高度
    total_height = sum(row_heights) + 107  # 107 是标题和边距的额外高度

    # 创建表格
    table_trace = go.Table(
        header=dict(
            values=list(table_df.columns),
            align="left",
            fill_color=ColorKit.LIGHT_GRAY,
            font=dict(color=ColorKit.BLACK, weight="bold"),
            line=dict(width=1, color=ColorKit.SLATE_GRAY),  # 单元格边框
        ),
        cells=dict(
            values=wrapped_values.T.values.tolist(),
            align="left",  # 单元格内容左对齐
            fill_color=ColorKit.WHITE,
            line=dict(width=1, color=ColorKit.SLATE_GRAY),  # 单元格边框
            font=dict(size=12),  # 字体大小
        ),
        columnwidth=column_width,  # 设置列宽
    )

    # 创建图表
    fig = go.Figure(data=[table_trace])
    fig.update_layout(
        width=width,
        height=total_height,  # 动态调整总高度
        title_text=title,
        title_x=0.5,
        margin=dict(t=50, b=20, l=20, r=20),  # 调整边距
        paper_bgcolor=ColorKit.WHITE,  # 整个图表区域背景色
    )

    # 返回 HTML div
    return_fig = plot(fig, include_plotlyjs=True, output_type="div")
    return return_fig


# 感谢蒜老板提供的代码，https://bbs.quantclass.cn/thread/60867
def draw_config(config_list):

    def custom_json_dumps(data, indent=2, compact_level=1):
        """智能压缩指定层级结构的JSON格式化函数"""
        filter_keys = ["funcs", "runtime_folder", "result_folder"]

        def _format(obj, level, compact_depth):
            if isinstance(obj, dict):
                items = []
                for k, v in obj.items():
                    if k in filter_keys:
                        continue
                    # 值格式化时继承当前层级
                    formatted = _format(v, level + 1, compact_depth)
                    items.append(f'"{k}" : {formatted}')
                joiner = f",\n{' ' * (level * indent)}" if level <= compact_depth else ", "
                return f"{{\n{' ' * (level * indent)}" + joiner.join(items) + f"\n{' ' * (level - 1) * indent}}}"

            elif isinstance(obj, list):
                if all(not isinstance(e, (dict, tuple, list)) for e in obj):
                    # 基础类型列表直接紧凑显示
                    return "[" + ", ".join(json.dumps(e, ensure_ascii=False) for e in obj) + "]"
                # 复杂结构列表智能换行
                items = []
                for e in obj:
                    formatted = _format(e, level + 1, compact_depth)
                    if isinstance(e, tuple):
                        formatted = formatted.replace("[", "(").replace("]", ")")
                    items.append(f"{formatted}")
                # 根据层级决定换行策略
                if level >= compact_depth:
                    return (
                        "[\n"
                        + f",\n".join([f"{' ' * (level * indent)}{item}" for item in items])
                        + f"\n{' ' * (level - 1) * indent}]"
                    )
                else:
                    return "[" + ", ".join(items) + "]"

            else:
                return json.dumps(obj, ensure_ascii=False)

        return _format(data, 1, compact_level)

    """
    生成可折叠的JSON配置HTML片段

    参数:
        strategy_configs: 策略配置列表（每个元素是一个字典）
        default_expanded: 是否默认展开所有配置块
        box_width: 配置块的宽度（CSS值，如"800px"或"90%"）
        max_height: 配置内容的最大高度（CSS值）
        title: 可选的配置区块标题

    返回:
        纯HTML片段字符串（可直接嵌入其他HTML）
    """
    if not config_list:
        return "<div>No configurations provided</div>"
    default_expanded = True
    box_width = "100%"
    max_height = "500px"
    # 生成每个配置块
    config_blocks = []
    for idx, strategy in enumerate(config_list):
        config_name = strategy.get("name", f"Configuration {idx}")
        json_str = (
            custom_json_dumps(strategy, indent=2, compact_level=2)
            .replace("true", "True")
            .replace("null", "None")
            .replace("false", "False")
        )
        # HTML特殊字符转义（防止XSS）
        json_str = (
            json_str.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

        config_blocks.append(
            f"""
        <div class="config-block" style="
            margin-bottom: 15px;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            overflow: hidden;
            width: {box_width};
            display: inline-block;   /* 关键修改：改为行内块元素 */
            vertical-align: top;      /* 顶部对齐 */
        ">
            <div 
                class="config-header" style="
                    padding: 10px 15px;
                    background-color: #f6f8fa;
                    cursor: pointer;
                    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                    font-weight: 600;
                    color: #24292e;
                    user-select: none;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;" >
                <span>{idx}. {config_name}</span>
                <span class="config-arrow" 
                      style="font-size: 12px;
                      margin-left: 10px; " >
                    {'▼' if default_expanded else '▶'}
                </span>
            </div>
            <pre class="config-content" style="
                margin: 0;
                padding: 15px;
                background-color: #ffffff;
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
                font-size: 12px;
                line-height: 1.5;
                overflow-x: auto;
                overflow-y: auto;
                max-height: {max_height};
                display: {'block' if default_expanded else 'none'};
                white-space: pre-wrap;
                word-wrap: break-word;
                border-top: 1px solid #e1e4e8;
            ">{json_str}</pre>
        </div>
        """
        )

    # 组合所有配置块
    return f"""
    <h2>策略配置</h2>
    <div class="strategy-configs-container" style="margin-bottom: 30px;width: 400px;white-space: nowrap;">
        {''.join(config_blocks)}
        <script>
            // 自动绑定所有配置块的点击事件
            document.addEventListener('DOMContentLoaded', function() {{
                document.querySelectorAll('.config-header').forEach(header => {{
                    // 初始化状态
                    const arrow = header.querySelector('.config-arrow');
                    const content = header.nextElementSibling;
                    arrow.textContent = content.style.display === 'none' ? '▶' : '▼';

                    // 单一事件监听器
                    header.addEventListener('click', function() {{
                        const isHidden = content.style.display === 'none';
                        content.style.display = isHidden ? 'block' : 'none';
                        arrow.textContent = isHidden ? '▼' : '▶';
                    }});
                }});
            }});

            // 全局控制函数（可选）
            function toggleAllConfigs(expand) {{
                document.querySelectorAll('.config-content').forEach(el => {{
                    el.style.display = expand ? 'block' : 'none';
                }});
                document.querySelectorAll('.config-header span').forEach(el => {{
                    el.textContent = expand ? '▼' : '▶';
                }});
            }}
        </script>
    </div>
    """


def merge_html(fig_path, fig_list):
    # 创建自定义HTML页面，嵌入fig对象的HTML内容
    # fmt: off
    icon = "data:image/x-icon;base64,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AP+VjQP/fZYU/z2wQv8puFH/KbhR/ym4Uf8puFH/KbhR/ym4Uf8puFH/KbhR/yq0VP8vl2//Rliw/+vq8//39/fh9/f3Ifb29jv39/f84cvv/65i3P+uYtz/rmLc/6dvlP+chCP/mYoD/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+Njwn/Uqcz/zC1TP8puFH/T6g1/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5KNBf9apC7/MrRL/ym4Uf8puFH/KbhR/ym4Uf8puFH/KbhR/ym4Uf8puFH/KrNV/7TZyP/39/f99/f3Qvj4+HL39/f/z6Xo/65i3P+uYtz/rmLc/65i2/+qabf/onpd/5qJDP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/l4wB/3SZG/9QqDX/hZMO/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+WjAL/eJgY/0irOv8rt0//KbhR/ym4Uf8puFH/KbhR/ym4Uf8puFH/KbhR/37Slv/39/f/9/f3e/f396L39vf/wIfj/65i3P+uYtz/rmLc/65i3P+uYtz/rWPU/6htof+cgyf/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5iLAf+TjQX/mIsB/5mLAP+ZiwD/mYsA/5mLAP+bhhf/pXR8/5uGGP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/4+PB/9lnyX/L7ZN/ym4Uf8puFH/KbhR/ym4Uf8puFH/KbhR/1fGdv/19vX/+Pj4q/f398T08fb/uXvg/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+rZsX/oHxR/5qIDv+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mKA/+ieGX/rWPW/6xmx/+gfE//mokM/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/gpQR/0KtPv8ttk7/KbhR/ym4Uf8puFH/KbhR/0fBaf/q8+3/9vb2zff399vt8vX/qa7o/6px3v+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rmPY/6dwkv+egTT/mYoD/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5yGHf+qabX/rmLc/65i3P+tZNH/pnCO/56CMv+ZigL/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5GOBv9mnyX/N7JG/ym4Uf8puFH/KbhR/0PAZv/e7+P/9/f35Pf39+Xq9fb/merx/5HR7f+kh+H/rmPc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i2/+sZsX/onhm/5mKAv+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/6R1dv+uYtr/rmLc/65i3P+uYtz/rmLb/6xnw/+jeGX/mooI/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+XjAL/iJEM/02pNv8quFD/KbhR/0G/ZP/X7d3/9/f38ff39+Tq9fb/mezy/4vr8f+N5PD/naTm/6tu3v+uY9z/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/6hsov+dgyv/mYoE/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/nYQo/61j1/+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtv/qG6d/5yDJv+Zigb/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5aMAv9qnSH/N7JG/0K/ZP/Y7d7/+Pj47/f399nt9vb/mu3y/4vr8f+L6/H/i+vx/5PJ7P+jjeL/rWXc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i2/+rZ77/o3ho/5uIEv+ZiwD/mYsA/5mLAP+aiA3/qG6c/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/6tmxP+jd2v/m4cU/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+YiwD/g5QQ/22vSP/i7+L/9/f34fj4+L/z9vb/ne3y/4vr8f+L6/H/i+vx/4vq8f+N4e//mrLo/6xp3f+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rWPW/6lrq/+dgjH/mYsB/5mLAP+gfE7/rWTR/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+tY9X/qmqw/56ANf+ZiwD/mYsA/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/6ScKP/x8ur/9/f3yfX19Z339/f/qO7y/4vr8f+L6/H/i+vx/4vr8f+L6/H/i+vx/5DU7f+khuH/rWbc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+sZcz/onle/5yFIv+obaD/rmLc/65i3P+uYtz/rmLc/65i3P+uY9z/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/61k0/+jdm7/m4cW/5mLAP+ZiwD/mYsA/5mLAP+ZiwD/mYsA/7ClPf/39vX/9/f3pfX19Wv39/f/v/H0/4vr8f+L6/H/i+vx/4vr8f+L6/H/i+vx/4vr8f+O3+//m63n/6l03/+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLa/6pqsf+tY9b/rmLc/65i3P+uYtz/rmLc/6ts3f+hleP/q27e/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtn/qGyk/59+SP+aiQj/mYsA/5mLAP+ZiwD/mYsA/8K7bv/39/f/9vb2dfb29jb39/f60e/3/3zk8v+L6/H/i+vx/4vr8f+L6/H/i+vx/4vr8f+L6/H/i+nx/5HU7f+jjOP/rmTc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/5+b5P+N5PD/ktDt/6aD4f+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/6xkzv+lcoX/mogR/5mLAP+ZiwD/mYsA/9zXrv/39/f88vLyPPb29h339/fV8fX3/zfC/P9h2PX/hejy/4vr8f+L6/H/i+vx/4vr8f+L6/H/i+vx/4vr8f+M5vD/m6vn/6pv3v+uY9z/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/qnDe/47g7/+L6/H/i+vx/4zl8P+cpub/qnDe/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+uYtz/q2i8/59/Pv+aiQr/n5EQ//Pz7f/39/fb9/f3H////wv39/eg9/f3/3vT+/8guv3/Qsr5/3Pf9P+K6vH/i+vx/4vr8f+L6/H/i+vx/4vr8f+L6/H/i+vx/5LN7P+ikeP/rWbd/65i3P+uYtz/rmLc/65i3P+uYtz/rmLc/65i3P+tZtz/mLfp/4vr8f+L6/H/i+vx/4vr8f+L6fH/lMnr/6OP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"
    # fmt: on
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <link rel="icon" href={icon} type="image/x-icon">
    <style>
        body {{
            background: {ColorKit.WHITE}
        }}
        .figure-container {{
            display: flex;
            flex-direction: column;
            position: relative;
        }}
        .electro {{
            position: absolute;  /* 绝对定位 */
            top: 50px;           /* 距离顶部距离，可调整 */
            left: 170px;          /* 距离左侧距离，可调整 */
            font-size: 32px;     /* 字体大小，可调整 */
            color: {ColorKit.LIGHT_GRAY}; /* 颜色和透明度，可调整 */
            pointer-events: none;
            z-index: 1000;
        }}
    </style>
    </head>
    <body>"""
    import base64

    electro = base64.b64decode("6YKi5LiN6KGM6YeP5YyW").decode("utf-8")
    for i, fig in enumerate(fig_list):
        # 将fig对象转换为HTML字符串
        if not isinstance(fig, str):
            fig = to_html(fig, full_html=False)
        if i == 0:
            html_content += f"""
            <div class="figure-container">
                {fig}
                <div class="electro">{electro}</div>
            </div>
            """
        else:
            html_content += f"""
            <div class="figure-container">
                {fig}
            </div>
            """
    html_content += "</body> </html>"

    # 保存自定义HTML页面
    with open(fig_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def save_performance(conf: BacktestConfig, **kwargs):
    # logger.debug(f'💾 保存回测结果到文件夹: {conf.get_result_folder()}')
    for name, df in kwargs.items():
        file_path = conf.get_result_folder() / f"{name}.csv"
        save_csv_safely(df, file_path, index=True)


def show_performance_plot(conf: BacktestConfig, select_results, equity_df, rtn, year_return, **kwargs):
    """
    绘制回测结果图表
    :param conf: 回测配置
    :param select_results: 选股结果
    :param equity_df: 回测结果净值数据
    :param rtn: 策略报告
    :param year_return: 分年收益率
    :param kwargs: 其他参数
    """
    # 添加指数数据
    for index_code, index_name in zip(["sh000300", "sh000852"], ["沪深300", "中证1000"]):
        index_path = conf.index_data_path / f"{index_code}.csv"
        if not index_path.exists():
            logger.warning(f"{index_name}({index_code})指数数据不存在，无法添加指数数据")
            continue
        index_df = import_index_data(index_path, [conf.start_date, conf.end_date])
        equity_df = pd.merge(left=equity_df, right=index_df[["交易日期", "指数涨跌幅"]], on=["交易日期"], how="left")
        equity_df[index_name + "指数"] = (equity_df["指数涨跌幅"] + 1).cumprod()
        del equity_df["指数涨跌幅"]

    logger.debug(
        f"""📈 策略评价 --------------------------------
{rtn}

📊 分年收益率 --------------------------------
{year_return}"""
    )
    logger.debug(f'💰 总手续费: ￥{equity_df["手续费"].sum():,.2f}\n')

    logger.info("开始绘制资金曲线...")

    # 生成画图数据字典，可以画出所有offset资金曲线以及各个offset资金曲线
    data_dict = {"资金曲线": "净值", "沪深300指数": "沪深300指数", "中证1000指数": "中证1000指数"}

    right_axis = {"最大回撤": "净值dd2here"}

    # 如果画资金曲线，同时也会画上回撤曲线
    date_start = equity_df["交易日期"].min().strftime("%Y/%m/%d")
    date_end = equity_df["交易日期"].max().strftime("%Y/%m/%d")
    ann_ret, max_dd, calmar = rtn.at["年化收益", 0], rtn.at["最大回撤", 0], rtn.at["年化收益/回撤比", 0]
    pic_title = f"年化收益:{ann_ret}  最大回撤:{max_dd}  收益回撤比:{calmar}  回测区间：{date_start} - {date_end}"
    pic_desc = ""
    # for stg in conf.strategy_list_raw:
    #     pic_desc += f'{stg["name"]}_{stg["hold_period"]}{stg["offset_list"]}_选{stg["select_num"]}_权{stg["cap_weight"]}_{stg["rebalance_time"]}+'

    fig_path = conf.get_result_folder() / "资金曲线.html"
    # 调用画图函数
    fig1 = draw_equity_curve_plotly(
        equity_df,
        data_dict=data_dict,
        date_col="交易日期",
        right_axis=right_axis,
        title=pic_title,
        desc=pic_desc[:-1],
        rtn_add=rtn,
        show_subplots=True,
    )

    # 获取每年选股最多的股票
    most_stock = get_most_stock_by_year(select_results) if not select_results.empty else pd.DataFrame()
    fig2 = draw_table(most_stock)

    figs = [fig1, fig2]

    # 绘制子策略的资金曲线图
    sub_equity_df = equity_df[["交易日期"]].copy()
    sub_data_dict = {}
    for col_name, col_series in kwargs.get("extra_equities", {}).items():
        sub_equity_df[col_name] = col_series
        sub_data_dict[col_name] = col_name
    if not sub_equity_df.empty and sub_data_dict:
        fig3 = draw_equity_curve_plotly(
            sub_equity_df, data_dict=sub_data_dict, date_col="交易日期", title="子策略资金曲线", desc=""
        )
        figs.append(fig3)
    # 绘制config
    figs.append(draw_config(conf.strategy_list_raw))
    # 绘制图片
    merge_html(fig_path, figs)

    import webbrowser
    import os
    import platform

    # 获取操作系统类型
    system = platform.system().lower()
    abs_path = os.path.abspath(str(fig_path))
    if system == "darwin":  # macOS
        os.system(f'open "{abs_path}"')
    elif system == "windows":  # Windows
        os.system(f"start {abs_path}")
    else:  # Linux 或其他系统
        webbrowser.open("file://" + abs_path)
