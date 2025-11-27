import matplotlib.pyplot as plt
import numpy as np
import random # 保留 import 以防其他部分依賴
def plot_routes(time_steps, agv_routes, total_time_steps):
    """根據 Gurobi 的最優路由結果繪製時間-空間圖。"""
    
    styles = {
        1: ('red', 'o', 'AGV1'),
        2: ('blue', 's', 'AGV2'),
        3: ('green', '^', 'AGV3'),
        4: ('orange', 'v', 'AGV4'),
        5: ('purple', 'D', 'AGV5')
    }
    
    plt.figure(figsize=(14, 8))
    
    for agv_id, route_array in agv_routes.items():
        # 長度修正
        if route_array.shape[0] != total_time_steps:
            print(f"⚠️ AGV {agv_id} 長度 {route_array.shape[0]} ≠ {total_time_steps}，自動補齊。")
            route_array = np.pad(route_array, (0, max(0, total_time_steps - route_array.shape[0])), mode='edge')[:total_time_steps]
        
        color, marker, label = styles.get(agv_id, ('black', 'x', f'AGV{agv_id}'))
        print(f"✅ Plotting {label}: len(time_steps)={len(time_steps)}, len(route_array)={len(route_array)}")
        
        plt.plot(
            time_steps, route_array,
            label=label, marker=marker, linestyle='-', color=color,
            markersize=5, linewidth=2, alpha=0.8
        )
    
    plt.xlabel('Time step [-]', fontsize=14)
    plt.ylabel('Node Number [-]', fontsize=14)
    plt.yticks(np.arange(1, 35, 5))
    plt.ylim(0, 35)
    plt.xlim(-1, total_time_steps)
    plt.xticks(np.arange(0, total_time_steps + 1, 5))
    plt.title('Optimal AGV Routing (Total Completion Time = 110)', fontsize=16)
    plt.legend(loc='upper right', ncol=3, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
