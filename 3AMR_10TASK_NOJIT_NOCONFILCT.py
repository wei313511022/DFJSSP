import gurobipy as gp
from gurobipy import GRB
import random
import matplotlib.pyplot as plt
import numpy as np

# ====================================================
# 繪圖函數定義 (保持不變)
# ====================================================
def plot_routes(time_steps, agv_routes, total_time_steps):
    """根據 Gurobi 的路由結果繪製時間-空間圖。"""
    
    styles = {
        1: ('red', 'o', 'AGV1'), 2: ('blue', 's', 'AGV2'), 3: ('green', '^', 'AGV3'), 
    }
    
    plt.figure(figsize=(14, 8)) 

    for agv_id, route in agv_routes.items():
        route_arr = np.asarray(route, dtype=float)
        
        # 簡易的長度檢查和調整
        if route_arr.shape[0] != total_time_steps:
             if route_arr.shape[0] < total_time_steps:
                 route_arr = np.pad(route_arr, (0, total_time_steps - route_arr.shape[0]), mode='edge')
             else:
                 route_arr = route_arr[:total_time_steps]
             
        color, marker, label = styles.get(agv_id, ('black', 'x', f'AGV{agv_id}')) 
        
        plt.plot(
            time_steps, route_arr,
            label=label, marker=marker, linestyle='-', color=color,
            markersize=5, linewidth=2, alpha=0.9
        )

    plt.xlabel('Time step [-]', fontsize=14)
    plt.ylabel('Node Number [-]', fontsize=14)
    plt.yticks(np.arange(0, 21, 5)) # 調整 Y 軸刻度以適應 20 個節點
    plt.ylim(0, 21)
    plt.xticks(np.arange(0, total_time_steps + 1, 5))
    plt.xlim(-1, total_time_steps)

    plt.title('VRP Routing (Min Total Completion Time - 3 AGVs / 20 Nodes)', fontsize=16)
    plt.legend(loc='upper right', ncol=3, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    outname = 'VRP_3AGV_20Node_MinTime.png'
    plt.savefig(outname, dpi=200)
    print(f"[繪圖通知] 圖表已生成並儲存為 '{outname}'。")
    plt.close()

# ====================================================
# --- 腳本開始： Gurobi 主邏輯 (20 節點, 30 時間步) ---
# ====================================================
if __name__ == "__main__":
    try:
        # --- 1. 定義集合與參數 (縮小規模) ---
        
        M_SET = range(1, 4)           # 3 輛 AGV 
        L_SET = range(1, 11)          # 10 個任務 
        N_SET = range(1, 21)          # *** 修正點：20 個節點 ***
        T_SET = range(30)             # *** 修正點：30 個時間步長 ***
        
        beta = 1 
        T_INDEX = T_SET 
        
        # AGV 初始位置 (確保在 1-20 範圍內)
        S_m = {1: 1, 2: 3, 3: 15} 
        
        # 任務資料 (隨機生成 1-20 範圍內的節點，確保有解)
        random.seed(10)
        Task_data = {}
        for i in L_SET:
            Task_data[i] = {'u_l': random.randint(1, 20), 'g_l': random.randint(1, 20)}

        # 模擬網絡結構: 鄰接節點 A_i (簡化為 4x5 網格結構以適應 20 節點)
        GRID_SIZE = 5 # 假設每行 5 個節點
        def get_adj_nodes_20(i):
            adj = {i}
            # 輔助函數: 將一維索引 i 轉換為 (row, col) 座標 (1-indexed)
            row = (i - 1) // GRID_SIZE 
            col = (i - 1) % GRID_SIZE
            
            # 上下左右四個方向
            if row > 0: adj.add(i - GRID_SIZE)       # 上
            if row < 4 - 1: adj.add(i + GRID_SIZE)   # 下
            if col > 0: adj.add(i - 1)               # 左
            if col < GRID_SIZE - 1: adj.add(i + 1)   # 右
            
            return list(adj)
        
        A_i = {i: get_adj_nodes_20(i) for i in N_SET}

        # --- 2. 建立 Gurobi 模型 ---
        model = gp.Model("VRP_20Node_30Time_Fast")
        
        # 設置較短的時限和較寬鬆的 Gap，以確保快速找到解
        model.setParam('TimeLimit', 60)  
        model.setParam('MIPGap', 0.05)     # 5% 容忍度，應該會非常快
        
        print(f"Gurobi 模型已建立 (20節點/30時間步)，目標：快速 VRP求解")

        # --- 3. 加入變數 (Variables) ---
        X = model.addVars(N_SET, N_SET, T_SET, M_SET, vtype=GRB.BINARY, name="X_move") 
        Y = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="Y_assign") 
        K = model.addVars(L_SET, T_SET, vtype=GRB.BINARY, name="K_not_pickup") 
        D = model.addVars(L_SET, T_SET, vtype=GRB.BINARY, name="D_not_delivery") 
        B = model.addVars(L_SET, T_SET, M_SET, vtype=GRB.BINARY, name="B_lin_pickup") 
        C = model.addVars(L_SET, T_SET, M_SET, vtype=GRB.BINARY, name="C_lin_delivery") 

        # --- 4. 設定目標函數 (最小化總完成時間) ---
        model.setObjective(
            beta * gp.quicksum(D[task, t] for task in L_SET for t in T_INDEX),
            GRB.MINIMIZE
        )

        # --- 5. 加入約束條件 (Constraints) ---

        # A. AGV 運動/流量約束
        model.addConstrs((gp.quicksum(X[i, j, t, agv] for j in A_i[i]) <= 1
                          for i in N_SET for t in T_SET for agv in M_SET), name="Constr3_MaxOneMove")
        model.addConstrs((gp.quicksum(X[i, j, t, agv] for i in N_SET for j in A_i[i]) == 1
                          for t in T_SET for agv in M_SET), name="Constr4_MustMoveOrStop")
        model.addConstrs((gp.quicksum(X[j, i, t, agv] for j in A_i[i]) == 
                          gp.quicksum(X[i, k, t+1, agv] for k in A_i[i])
                          for i in N_SET for t in T_SET[:-1] for agv in M_SET), name="Constr5_FlowConservation")
        
        # 初始位置約束 (t=0)
        for agv in M_SET:
            start_node = S_m[agv]
            model.addConstr(gp.quicksum(X[start_node, j, 0, agv] for j in A_i[start_node]) == 1, name=f"Constr9_StartNode_{agv}")
            model.addConstrs((gp.quicksum(X[i, j, 0, agv] for j in A_i[i]) == 0
                              for i in N_SET if i != start_node), name=f"Constr10_OtherNodes_{agv}")


        # B. 衝突無限制約束 (已刪除約束 6, 7, 8)


        # C. 任務分配約束 (論文 Eq. (33))
        model.addConstrs((gp.quicksum(Y[task, agv] for agv in M_SET) == 1
                          for task in L_SET), name="Constr33_AssignOneVehicle")


        # D. 拾取/交付狀態變數 (K, D) 及其線性化輔助變數 (B, C)
        # 初始狀態 (論文 Eq. (17))
        model.addConstrs((K[task, 0] == 1 for task in L_SET), name="Constr17_K_initial")
        model.addConstrs((D[task, 0] == 1 for task in L_SET), name="Constr17_D_initial")

        for task in L_SET:
            u_l, g_l = Task_data[task]['u_l'], Task_data[task]['g_l']
            for t in T_SET[:-1]: # 使用 t
                
                # 線性化約束 (使用 t)
                for agv in M_SET:
                    model.addConstr(B[task, t, agv] <= Y[task, agv], name=f"Lin_B1_{task}_{t}_{agv}")
                    model.addConstr(B[task, t, agv] <= X[u_l, u_l, t, agv], name=f"Lin_B2_{task}_{t}_{agv}")
                    model.addConstr(B[task, t, agv] >= Y[task, agv] + X[u_l, u_l, t, agv] - 1, name=f"Lin_B3_{task}_{t}_{agv}")
                    model.addConstr(C[task, t, agv] <= Y[task, agv], name=f"Lin_C1_{task}_{t}_{agv}")
                    model.addConstr(C[task, t, agv] <= X[g_l, g_l, t, agv], name=f"Lin_C2_{task}_{t}_{agv}")
                    model.addConstr(C[task, t, agv] >= Y[task, agv] + X[g_l, g_l, t, agv] - 1, name=f"Lin_C3_{task}_{t}_{agv}")

                # 狀態轉換約束 (使用 t)
                model.addConstr(K[task, t] - gp.quicksum(B[task, t, agv_q] for agv_q in M_SET) <= K[task, t+1], name=f"Constr13_K_Update1_{task}_{t}")
                model.addConstr(gp.quicksum(B[task, t, agv_q] for agv_q in M_SET) + K[task, t+1] <= 1, name=f"Constr14_K_Update2_{task}_{t}")
                model.addConstr(D[task, t] - gp.quicksum(C[task, t, agv_q] for agv_q in M_SET) <= D[task, t+1], name=f"Constr15_D_Update1_{task}_{t}")
                model.addConstr(gp.quicksum(C[task, t, agv_q] for agv_q in M_SET) + D[task, t+1] <= 1, name=f"Constr16_D_Update2_{task}_{t}")

                # 狀態非增性 & 順序 (論文 Eq. (18))
                model.addConstr(K[task, t] <= D[task, t], name=f"Constr18_PickupBeforeDelivery_{task}_{t}")
                model.addConstr(K[task, t+1] <= K[task, t], name=f"Constr18_K_NonIncrease_{task}_{t}")
                model.addConstr(D[task, t+1] <= D[task, t], name=f"Constr18_D_NonIncrease_{task}_{t}")


        # --- 6. 求解模型 (Solve Model) ---
        print("\n開始求解模型...")
        model.update()
        model.optimize()

        # --- 7. 輸出結果 (Analyze Results) ---
        
        all_agv_routes = {}
        
        print("\n==========================================================")
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            print(f"目標函數值 (總完成時間): {model.objVal:.2f}")
        else:
             print(f"❌ 模型求解失敗或無可行解。Gurobi 狀態碼: {model.status}") 

        print("==========================================================")
            
        # 輸出 AGV 分配結果
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            print("\n📌 任務分配 (Task Assignment):")
            for task in L_SET:
                for agv in M_SET:
                    if Y[task, agv].X > 0.5:
                        print(f"  - 任務 {task} (拾取點 {Task_data[task]['u_l']} -> 交付點 {Task_data[task]['g_l']}) 分配給 AGV {agv}")

            # 輸出 AGV 路由結果 (全部時間步長)
            time_steps_output = list(T_SET)
            TOTAL_TIME_STEPS = len(T_SET)

            print(f"\n📌 AGV 路由結果 (t=0到{TOTAL_TIME_STEPS-1}, 共 {TOTAL_TIME_STEPS} 個時間步):")
            for agv in M_SET:
                route = []
                current_node = S_m[agv]
                for t in T_SET: 
                    for i in N_SET:
                        if any(X[i, j, t, agv].X > 0.5 for j in A_i[i]):
                            current_node = i
                            break
                    route.append(current_node)
                all_agv_routes[agv] = route
                print(f"  - AGV {agv} 路由 (t=0到{TOTAL_TIME_STEPS-1}): {route}")

            # 呼叫繪圖函數
            plot_routes(time_steps_output, all_agv_routes, TOTAL_TIME_STEPS)
        
    except gp.GurobiError as e:
        print(f"\n❌ Gurobi 錯誤代碼 {e.errno}: {e}")
        print("請確認 Gurobi 已安裝並取得有效授權。")
    except Exception as e:
        print(f"\n❌ 發生一般錯誤: {e}")