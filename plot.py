import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

# 设置专业风格的绘图参数
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'DejaVu Sans'  # 使用无衬线字体
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.figsize'] = (14, 8)
mpl.rcParams['figure.dpi'] = 100

# 专业配色方案（来自Tableau）
BATCH_COLORS = {
    0: '#4E79A7',  # 深蓝
    1: '#F28E2B',  # 橙色
    2: '#E15759',  # 红色
    3: '#76B7B2',  # 青绿
    4: '#59A14F',  # 绿色
    5: '#EDC948',  # 黄色
    6: '#B07AA1',  # 紫色
    7: '#FF9DA7',  # 粉红
    8: '#9C755F',  # 棕色
    9: '#BAB0AC',  # 灰色,
    10: '#499894', # 青色
    11: '#A25050'  # 深红
}

# 请求事件专用颜色
REQUEST_COLORS = {
    'added': '#2E8B57',  # 海绿色 - 请求到达
    'finished': '#DC143C'  # 深红色 - 请求完成
}

# 解析日志文件的函数 - 增强：支持汇总请求条目与阶段标注
def parse_log_file(filename, log_type):
    """
    返回:
      events: dict batch_id -> list of (event_type, timestamp, batch_size, stage)
              event_type: 'compute_start','compute_end','trans_start','trans_end','serial_start','serial_end','back_to_head','recv'
              stage: None or string like 'prefill'/'decode' 等
      request_events: dict timestamp -> list of tuples:
              - ('added', req_id) or ('finished', req_id)  (原先逐条)
              - ('added_count', count) or ('finished_count', count) (汇总条目)
    """
    events = defaultdict(list)
    request_events = defaultdict(list)  # 专门存储请求事件
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # --- 新增：匹配汇总请求条目，例如 "1 requests finished at 1761024545.164717"
            m_req_count = re.match(r'(\d+)\s+requests\s+(finished|added)\s+at\s+([\d.]+)$', line)
            if m_req_count:
                count = int(m_req_count.group(1))
                action = m_req_count.group(2)  # 'finished' 或 'added'
                ts = float(m_req_count.group(3))
                key = f'{action}_count'  # 'finished_count' 或 'added_count'
                request_events[ts].append((key, count))
                continue

            # --- 兼容原先的逐条 request 日志: "request <id> is added at <ts>" 或 "request <id> finished at <ts>"
            request_added_match = re.match(r'request\s+([a-f0-9]+)\s+is added at\s+([\d.]+)$', line)
            request_finished_match = re.match(r'request\s+([a-f0-9]+)\s+finished at\s+([\d.]+)$', line)
            if request_added_match:
                request_id = request_added_match.group(1)
                timestamp = float(request_added_match.group(2))
                request_events[timestamp].append(('added', request_id))
                continue
            if request_finished_match:
                request_id = request_finished_match.group(1)
                timestamp = float(request_finished_match.group(2))
                request_events[timestamp].append(('finished', request_id))
                continue

            # --- 匹配常规事件行: "0 1 compute starts (prefill) at 1761024538.0907214"
            # 支持可选的 batch_size 和可选的阶段括号
            match = re.match(r'(\d+)(?:\s+(\d+))?\s+(.+?)\s+at\s+([\d.]+)$', line)
            if not match:
                continue
                
            batch_id = int(match.group(1))
            batch_size = match.group(2)
            event_desc = match.group(3).strip()
            timestamp = float(match.group(4))

            # 尝试提取阶段注释，例如 "compute starts (prefill)" -> stage='prefill'
            stage_match = re.search(r'\(([^)]+)\)', event_desc)
            stage = stage_match.group(1) if stage_match else None

            # 规范化 batch_size
            if batch_size:
                batch_size = int(batch_size)
            else:
                # 尝试从事件描述中提取 batch size（若像 "8 compute starts" 的情况）
                size_match = re.search(r'(\d+)\s+compute starts', event_desc)
                if size_match:
                    batch_size = int(size_match.group(1))

            # 提取事件类型
            event_type = None
            if "compute starts" in event_desc:
                event_type = 'compute_start'
            elif "compute ends" in event_desc:
                event_type = 'compute_end'
            elif "trans starts" in event_desc:
                event_type = 'trans_start'
            elif "trans ends" in event_desc:
                event_type = 'trans_end'
            elif "serialization starts" in event_desc:
                event_type = 'serial_start'
            elif "serialization ends" in event_desc:
                event_type = 'serial_end'
            elif "back to head" in event_desc:
                event_type = 'back_to_head'
            elif "recv" in event_desc:
                event_type = 'recv'
            else:
                continue  # 忽略未知事件类型

            # 存储为四元组，stage 可能为 None
            events[batch_id].append((event_type, timestamp, batch_size or 0, stage))
            
    return events, request_events

# 解析日志文件
print("解析server1日志...")
server1_events, server1_requests = parse_log_file('/home/emnets-2/gxq/MoLink/server1.log', 'server1')
print("解析server2日志...")
server2_events, server2_requests = parse_log_file('/home/emnets-2/gxq/MoLink/server2.log', 'server2')

# 提取所有时间戳用于归一化
all_timestamps = []
for batch_id, events in server1_events.items():
    print(f"Server1 Batch {batch_id} 事件:")
    for event_type, ts, size, stage in events:
        print(f"  {event_type} ({stage}) at {ts}")
        all_timestamps.append(ts)

for batch_id, events in server2_events.items():
    print(f"Server2 Batch {batch_id} 事件:")
    for event_type, ts, size, stage in events:
        print(f"  {event_type} ({stage}) at {ts}")
        all_timestamps.append(ts)

# 添加请求事件的时间戳
for ts, events in server1_requests.items():
    for tup in events:
        # tup 可能是 ('added', req_id) 或 ('finished', req_id) 或 ('finished_count', count)
        print(f"Server1 Request event {tup} at {ts}")
        all_timestamps.append(ts)

for ts, events in server2_requests.items():
    for tup in events:
        print(f"Server2 Request event {tup} at {ts}")
        all_timestamps.append(ts)

if not all_timestamps:
    print("错误: 没有解析到任何事件!")
    exit(1)

base_time = min(all_timestamps)
print(f"基准时间: {base_time}")

# 归一化时间戳函数
def normalize(ts):
    return ts - base_time

# 创建绘图
fig, ax = plt.subplots(figsize=(16, 9))
fig.set_facecolor('#F8F9FA')  # 浅灰色背景
ax.set_facecolor('#FFFFFF')   # 白色轴背景

# 资源层级定义 - 增加间距
RESOURCE_LEVELS = {
    'Server1 Compute': 4.5,
    'Network Transfer1': 3.0,
    'Server2 Compute': 1.5,
    'Network Transfer2': 1.0,
}

# 请求事件位置
REQUEST_LEVEL = 5.0  # Server1 Compute上方

# 线宽和透明度设置 - 改为条高度（数据单位）
COMPUTE_BAR_HEIGHT = 0.35  # 计算条高度（y轴单位）
TRANSFER_BAR_HEIGHT = 0.25  # 传输条高度（y轴单位）
ALPHA = 0.9

# 设置要显示的时间范围（使用原始时间戳）
# 根据您提供的数据调整
start_time = 1769267526.776886
end_time = 1769267527.7092068



# 归一化时间范围
norm_start = normalize(start_time)
norm_end = normalize(end_time)
print(f"显示时间范围: {norm_start:.6f} 到 {norm_end:.6f} (相对基准时间)")

# 存储批注对象用于调整位置
annotations = []
legend_handles = []  # 用于自定义图例
found_transfers = []  # 存储找到的传输事件

# 绘制请求事件函数 - 支持汇总数量条目
def draw_request_events(request_events):
    request_counts = defaultdict(lambda: defaultdict(int))  # 按时间戳和事件类型计数
    
    # 统计每个时间戳的事件数量
    for ts, events in request_events.items():
        for ev in events:
            # ev 可能是 ('added', req_id) 或 ('finished', req_id) 或 ('added_count', count) ...
            etype = ev[0]
            if etype.endswith('_count'):
                # ('finished_count', count)
                action = etype.replace('_count', '')
                count = ev[1]
                request_counts[ts][action] += int(count)
            else:
                # ('added', req_id) 或 ('finished', req_id)
                action = etype
                request_counts[ts][action] += 1
    
    # 绘制事件
    for ts, event_counts in request_counts.items():
        norm_ts = normalize(ts)
        
        # 检查是否在时间范围内
        if norm_start <= norm_ts <= norm_end:
            for event_type, count in event_counts.items():
                color = REQUEST_COLORS.get(event_type, '#000000')
                event_name = "arrives" if event_type == "added" else "finishes"
                
                # 绘制箭头指向Server1 Compute
                arrow = FancyArrowPatch(
                    (norm_ts, REQUEST_LEVEL), (norm_ts, RESOURCE_LEVELS['Server1 Compute'] + 0.2),
                    arrowstyle='->', mutation_scale=15, color=color, linewidth=2, alpha=0.8
                )
                ax.add_patch(arrow)
                
                # 添加文本标注
                text = f"{count} req {event_name}"
                ann = ax.text(
                    norm_ts, REQUEST_LEVEL + 0.15, text,
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=color,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, boxstyle='round,pad=0.3')
                )
                annotations.append(ann)
                
                # 添加图例句柄（只添加一次）
                if event_type not in [h[0] for h in legend_handles if isinstance(h[0], str)]:
                    legend_handles.append((event_type, plt.Line2D([0], [0], color=color, lw=3, marker='o', markersize=8)))

# 绘制计算段函数 - 支持阶段并用 hatch 区分
def draw_compute_segments(events, resource_level, is_server1=True):
    for batch_id, batch_events in events.items():
        # 收集所有计算开始和结束事件，按阶段配对
        compute_starts = [(ts, size, stage) for etype, ts, size, stage in batch_events if etype == 'compute_start']
        compute_ends = [(ts, stage) for etype, ts, _, stage in batch_events if etype == 'compute_end']
        
        # 按时间排序以确保正确匹配
        compute_starts.sort(key=lambda x: x[0])
        compute_ends.sort(key=lambda x: x[0])
        
        # 对于可能存在的阶段匹配逻辑，我们按出现顺序配对：第 i 个 start 对应第 i 个 end（如果阶段相同则最好）
        min_count = min(len(compute_starts), len(compute_ends))
        if len(compute_starts) != len(compute_ends):
            print(f"警告: Batch {batch_id} 的计算开始({len(compute_starts)})和结束({len(compute_ends)})事件数量不匹配，取最小值 {min_count} 进行匹配")
            compute_starts = compute_starts[:min_count]
            compute_ends = compute_ends[:min_count]
        
        for (start, batch_size, start_stage), (end, end_stage) in zip(compute_starts, compute_ends):
            # 优先使用 start_stage，如果为空使用 end_stage
            stage = start_stage or end_stage
            norm_start_ts = normalize(start)
            norm_end_ts = normalize(end)
            
            # 检查是否在时间范围内
            if norm_end_ts > norm_start and norm_start_ts < norm_end:
                # 计算实际显示的起始点和结束点
                display_start = max(norm_start_ts, norm_start)
                display_end = min(norm_end_ts, norm_end)
                
                # 计算持续时间（毫秒）
                duration_ms = (end - start) * 1000
                
                # 选择颜色（保持原色）
                color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                
                # 计算矩形高度和位置（用 Rectangle 来支持 hatch）
                bar_height = COMPUTE_BAR_HEIGHT
                y_bottom = resource_level - bar_height / 2.0
                
                rect = Rectangle(
                    (display_start, y_bottom),
                    display_end - display_start,
                    bar_height,
                    facecolor=color,
                    edgecolor=color,
                    alpha=ALPHA,
                    linewidth=0.5,
                    zorder=2
                )
                
                # 根据阶段设置 hatch（不改变颜色）
                if stage:
                    stage_lower = stage.lower()
                    if 'prefill' in stage_lower:
                        rect.set_hatch('//')   # prefill 使用斜线
                    elif 'decode' in stage_lower:
                        rect.set_hatch('..')   # decode 使用点状（注意: 点样式在 matplotlib 使用中显示可能较稀疏）
                    else:
                        # 其他阶段可以使用横线作为示例，但若不想改变就不设置 hatch
                        pass
                
                ax.add_patch(rect)
                
                # 添加图例句柄（只添加一次）
                if batch_id not in [h[0] for h in legend_handles if isinstance(h[0], int)]:
                    legend_handles.append((batch_id, plt.Line2D([0], [0], color=color, lw=4, alpha=ALPHA)))
                
                # 修改：将文本标注放在计算段下方或者上方
                mid_x = (display_start + display_end) / 2
                text = f"bs={batch_size}\n{duration_ms:.1f}ms\n{stage or ''}"
                
                if is_server1:
                    ann_y = resource_level - bar_height/2 - 0.15
                else:
                    ann_y = resource_level + bar_height/2 + 0.15
                
                ann = ax.text(
                    mid_x, ann_y, text,
                    ha='center', va='center', fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                )
                annotations.append(ann)

# 绘制传输段函数 - 精确匹配您提供的格式 （保持原逻辑，只是用 Rectangle 绘制以保证样式一致）
def draw_transfer_segments():
    global found_transfers
    
    print("\n绘制中间结果传输 (server1 -> server2): trans_start to recv")
    for batch_id in set(list(server1_events.keys()) + list(server2_events.keys())):
        # 获取server1的trans_start事件
        s1_trans_starts = [ts for etype, ts, _, _ in server1_events.get(batch_id, []) 
                          if etype == 'trans_start']
        s1_trans_starts.sort()
        
        # 获取server2的recv事件
        s2_recvs = [ts for etype, ts, _, _ in server2_events.get(batch_id, []) 
                   if etype == 'recv']
        s2_recvs.sort()
        
        print(f"Batch {batch_id}: trans_starts={len(s1_trans_starts)}, recvs={len(s2_recvs)}")
        
        # 确保有匹配的事件
        if s1_trans_starts and s2_recvs:
            # 按顺序匹配事件
            for i in range(min(len(s1_trans_starts), len(s2_recvs))):
                start = s1_trans_starts[i]
                end = s2_recvs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                print(f"  中间传输 {i+1}: {start} -> {end} (原始时间)")
                print(f"  归一化后: {norm_start_ts:.6f} -> {norm_end_ts:.6f}")
                
                # 检查是否在时间范围内
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    # 计算实际显示的起始点和结束点
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    # 计算持续时间（毫秒）
                    duration_ms = (end - start) * 1000
                    
                    # 选择颜色
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    print(f"  在时间范围内! 绘制线段: {display_start:.6f} -> {display_end:.6f}")
                    
                    # 使用 Rectangle 绘制传输段（保持原样式）
                    bar_height = TRANSFER_BAR_HEIGHT
                    y_bottom = RESOURCE_LEVELS['Network Transfer1'] - bar_height / 2.0
                    rect = Rectangle(
                        (display_start, y_bottom),
                        display_end - display_start,
                        bar_height,
                        facecolor=color,
                        edgecolor=color,
                        alpha=ALPHA,
                        linewidth=0.5,
                        zorder=1
                    )
                    ax.add_patch(rect)
                    
                    # 添加文本标注 - 放在线段下方
                    mid_x = (display_start + display_end) / 2
                    text = f"To Server2\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network Transfer1'] - bar_height/2 - 0.15
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("intermediate", batch_id, start, end))
    
    print("\n绘制结果返回传输 (server2 -> server1): trans_start to back_to_head")
    for batch_id in set(list(server1_events.keys()) + list(server2_events.keys())):
        # 获取server2的trans_start事件
        s2_trans_starts = [ts for etype, ts, _, _ in server2_events.get(batch_id, []) 
                          if etype == 'trans_start']
        s2_trans_starts.sort()
        
        # 获取server1的back_to_head事件
        s1_backs = [ts for etype, ts, _, _ in server1_events.get(batch_id, []) 
                   if etype == 'back_to_head']
        s1_backs.sort()
        
        print(f"Batch {batch_id}: trans_starts={len(s2_trans_starts)}, back_to_heads={len(s1_backs)}")
        
        # 确保有匹配的事件
        if s2_trans_starts and s1_backs:
            # 按顺序匹配事件
            for i in range(min(len(s2_trans_starts), len(s1_backs))):
                start = s2_trans_starts[i]
                end = s1_backs[i]
                
                norm_start_ts = normalize(start)
                norm_end_ts = normalize(end)
                
                print(f"  结果传输 {i+1}: {start} -> {end} (原始时间)")
                print(f"  归一化后: {norm_start_ts:.6f} -> {norm_end_ts:.6f}")
                
                # 检查是否在时间范围内
                if norm_end_ts > norm_start and norm_start_ts < norm_end:
                    # 计算实际显示的起始点和结束点
                    display_start = max(norm_start_ts, norm_start)
                    display_end = min(norm_end_ts, norm_end)
                    
                    # 计算持续时间（毫秒）
                    duration_ms = (end - start) * 1000
                    
                    # 选择颜色
                    color = BATCH_COLORS.get(batch_id % len(BATCH_COLORS), '#1f77b4')
                    
                    print(f"  在时间范围内! 绘制线段: {display_start:.6f} -> {display_end:.6f}")
                    
                    # 绘制（使用 Rectangle），并用不同的线型区分（通过 edge linestyle 无法直接设置在 Rectangle 上，
                    # 我们保持 facecolor，但为了视觉差异可以降低 alpha 或稍微改变 edge）
                    bar_height = TRANSFER_BAR_HEIGHT
                    y_bottom = RESOURCE_LEVELS['Network Transfer2'] - bar_height / 2.0
                    rect = Rectangle(
                        (display_start, y_bottom),
                        display_end - display_start,
                        bar_height,
                        facecolor=color,
                        edgecolor=color,
                        alpha=ALPHA * 0.85,
                        linewidth=0.5,
                        zorder=1
                    )
                    ax.add_patch(rect)
                    
                    # 添加文本标注 - 放在线段下方
                    mid_x = (display_start + display_end) / 2
                    text = f"To Server1\n{duration_ms:.1f}ms"
                    ann_y = RESOURCE_LEVELS['Network Transfer2'] - bar_height/2 - 0.15
                    ann = ax.text(
                        mid_x, ann_y, text,
                        ha='center', va='center', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                    )
                    annotations.append(ann)
                    found_transfers.append(("result", batch_id, start, end))

# 绘制所有部分
print("\n绘制Server1计算段...")
draw_compute_segments(server1_events, RESOURCE_LEVELS['Server1 Compute'])
print("绘制Server2计算段...")
draw_compute_segments(server2_events, RESOURCE_LEVELS['Server2 Compute'], is_server1=False)
print("绘制网络传输段...")
draw_transfer_segments()
print("绘制请求事件...")
draw_request_events(server1_requests)

if not found_transfers:
    print("警告: 没有找到任何网络传输事件!")
else:
    print(f"成功绘制 {len(found_transfers)} 个传输事件")

# 设置图表属性
ax.set_yticks(list(RESOURCE_LEVELS.values()) + [REQUEST_LEVEL])
ax.set_yticklabels(list(RESOURCE_LEVELS.keys()) + ['Request Events'], fontsize=12, fontweight='bold')
ax.set_ylabel('Resources', fontsize=12, fontweight='bold')
ax.set_xlabel(f'Time (seconds from base time {base_time:.6f})', fontsize=11)
ax.set_title(f'Distributed Computing Timeline', fontsize=14, fontweight='bold', pad=15)

# 设置x轴范围
ax.set_xlim(norm_start, norm_end)

# 设置y轴范围以包含请求事件和标注
ax.set_ylim(0.3, 5.5)  # 稍微扩大y轴范围以容纳下方的标注

# 添加时间刻度线
ax.grid(True, axis='x', linestyle='--', alpha=0.6)

# 添加资源分隔线
for y in RESOURCE_LEVELS.values():
    ax.axhline(y=y, color='gray', alpha=0.3, linewidth=0.5)

# 添加请求事件水平线
ax.axhline(y=REQUEST_LEVEL, color='gray', alpha=0.3, linewidth=0.5, linestyle=':')

# 创建专业图例
if legend_handles:
    # 分离batch和请求事件的图例
    batch_handles = [h[1] for h in legend_handles if isinstance(h[0], int)]
    batch_labels = [f'Batch {h[0]}' for h in legend_handles if isinstance(h[0], int)]
    
    request_handles = [h[1] for h in legend_handles if isinstance(h[0], str)]
    request_labels = [f'Request {h[0].title()}' for h in legend_handles if isinstance(h[0], str)]
    
    # Batch颜色图例
    if batch_handles:
        batch_legend = plt.legend(
            batch_handles,
            batch_labels,
            title='Batch ID',
            loc='upper left',
            bbox_to_anchor=(0.01, 0.99),
            frameon=True,
            framealpha=0.9,
            edgecolor='#CCCCCC'
        )
        ax.add_artist(batch_legend)
    
    # 请求事件图例
    if request_handles:
        request_legend = plt.legend(
            request_handles,
            request_labels,
            title='Request Events',
            loc='upper left',
            bbox_to_anchor=(0.01, 0.85),
            frameon=True,
            framealpha=0.9,
            edgecolor='#CCCCCC'
        )
        ax.add_artist(request_legend)

# 添加时间范围标记
ax.text(
    0.5, -0.12, 
    f'Time Range: {start_time:.6f} - {end_time:.6f} | Base Time: {base_time:.6f}',
    transform=ax.transAxes,
    ha='center',
    fontsize=9,
    color='#555555'
)

# 添加边框
for spine in ax.spines.values():
    spine.set_edgecolor('#DDDDDD')
    spine.set_linewidth=0.8

# 调整标注位置避免重叠
def adjust_annotations(annotations, min_distance=0.02):
    """调整标注位置避免重叠"""
    # 按x坐标排序
    annotations.sort(key=lambda ann: ann.get_position()[0])
    
    # 检查并调整重叠
    for i in range(1, len(annotations)):
        prev = annotations[i-1]
        curr = annotations[i]
        prev_pos = prev.get_position()
        curr_pos = curr.get_position()
        
        # 检查x坐标是否太近
        if abs(curr_pos[0] - prev_pos[0]) < min_distance:
            # 垂直偏移调整
            offset = min_distance - abs(curr_pos[0] - prev_pos[0])
            new_y = curr_pos[1] - offset * 0.5
            curr.set_position((curr_pos[0], new_y))

# 应用标注调整
if annotations:
    adjust_annotations(annotations)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.15, top=0.92, left=0.08, right=0.95)

# 保存图像（可选）
plt.savefig('/home/emnets-2/gxq/MoLink/plot/distributed_timeline.pdf', bbox_inches='tight', dpi=300)
plt.show()
