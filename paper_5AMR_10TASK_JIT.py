import gurobipy as gp
from gurobipy import GRB
import random
import matplotlib.pyplot as plt
import numpy as np

def plot_routes(time_steps, agv_routes, total_time_steps):
    """根據 Gurobi 的路由結果繪製時間-空間圖（穩健版本）。"""
    styles = {
        1: ('red', 'o', 'AGV1'),
        2: ('blue', 's', 'AGV2'),
        3: ('green', '^', 'AGV3'),
        4: ('orange', 'v', 'AGV4'),
        5: ('purple', 'D', 'AGV5')
    }

    plt.figure(figsize=(14, 8))

    for agv_id, route in agv_routes.items():
        # 轉為 numpy array（確保是數字）
        route_arr = np.asarray(route, dtype=float)

        # 若長度不符，自動補齊或截斷（保證與 time_steps 對應）
        if route_arr.shape[0] != total_time_steps:
            print(f"⚠️ AGV{agv_id} 長度 {route_arr.shape[0]} ≠ {total_time_steps}，將自動補齊/截斷。")
            if route_arr.shape[0] < total_time_steps:
                route_arr = np.pad(route_arr, (0, total_time_steps - route_arr.shape[0]), mode='edge')
            else:
                route_arr = route_arr[:total_time_steps]

        color, marker, label = styles.get(agv_id, ('black', 'x', f'AGV{agv_id}'))

        # debug 輸出（可移除）
        print(f"Plotting {label}: len(time_steps)={len(time_steps)}, len(route)={route_arr.shape[0]}")

        plt.plot(
            time_steps, route_arr,
            label=label, marker=marker, linestyle='-', color=color,
            markersize=5, linewidth=2, alpha=0.9
        )

    plt.xlabel('Time step [-]', fontsize=14)
    plt.ylabel('Node Number [-]', fontsize=14)
    plt.yticks(np.arange(0, 35, 5))
    plt.ylim(0, 32)
    plt.xticks(np.arange(0, total_time_steps + 1, 5))
    plt.xlim(-1, total_time_steps)
    plt.title('paper_5AMR_10TASK_JIT')
    plt.legend(loc='upper right', ncol=3, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)

    outname = 'paper_5AMR_10TASK_NOJIT.png'
    plt.savefig(outname, dpi=200)
    plt.close()
    print(f"[繪圖通知] 圖表已生成並儲存為 '{outname}'。")

# --- 腳本開始：直接運行所有邏輯 ---
if __name__ == "__main__":
    try:
        # --- 1. 定義集合與參數 (論文設定) ---
        
        M_SET = range(1, 6)           # 5 輛 AGV 
        L_SET = range(1, 11)          # 10 個任務 
        N_SET = range(1, 32)          # 31 個節點 
        T_SET = range(45)             # 45 個時間步長 
        
        alpha = 1  
        beta = 1   
        T_INDEX = T_SET # 用于目标函数和惩罚项计算
        
        # AGV 初始位置 (論文 Table I)
        S_m = {1: 1, 2: 3, 3: 25, 4: 12, 5: 7} 
        
        # 任務資料 (論文 Table II 數據)
        Task_data = {
            1: {'u_l': 6, 'g_l': 11, 'P_l^p': 11, 'P_l^d': 21}, 2: {'u_l': 8, 'g_l': 13, 'P_l^p': 14, 'P_l^d': 24}, 
            3: {'u_l': 30, 'g_l': 4, 'P_l^p': 16, 'P_l^d': 26}, 4: {'u_l': 17, 'g_l': 29, 'P_l^p': 18, 'P_l^d': 28}, 
            5: {'u_l': 12, 'g_l': 17, 'P_l^p': 21, 'P_l^d': 31}, 6: {'u_l': 20, 'g_l': 25, 'P_l^p': 23, 'P_l^d': 33}, 
            7: {'u_l': 24, 'g_l': 21, 'P_l^p': 24, 'P_l^d': 34}, 8: {'u_l': 3, 'g_l': 16, 'P_l^p': 25, 'P_l^d': 35}, 
            9: {'u_l': 5, 'g_l': 10, 'P_l^p': 27, 'P_l^d': 37}, 10: {'u_l': 26, 'g_l': 8, 'P_l^p': 30, 'P_l^d': 40} 
        }

        # 模擬網絡結構: 鄰接節點 A_i (論文 Fig. 6 的結構)
        ADJACENCY = {
            1: [1, 31, 2], 2: [2, 18, 3, 1], 3: [3, 4, 2], 4: [4, 17, 5, 3], 5: [5, 19, 6, 4], 
            6: [6, 15, 7, 5], 7: [7, 8, 6, 14], 8: [8, 9, 7, 22], 9: [9, 10, 8], 
            10: [10, 11, 9], 11: [11, 10, 12], 12: [12, 13, 11], 13: [13, 14, 12, 22], 
            14: [14, 15, 13, 7], 15: [15, 16, 14, 6], 16: [16, 17, 15], 17: [17, 4, 16], 
            18: [18, 19, 31, 2], 19: [19, 20, 5, 18], 20: [20, 21, 19, 29], 21: [21, 22, 20, 28], 
            22: [22, 23, 21, 8, 13], 23: [23, 24, 22], 24: [24, 25, 23], 25: [25, 24, 26], 
            26: [26, 27, 25, 27], 27: [27, 28, 26], 28: [28, 29, 27, 21], 29: [29, 30, 28, 20], 
            30: [30, 31, 29], 31: [31, 1, 18, 30]
        }
        A_i = {i: list(set(ADJACENCY[i] + [i])) for i in N_SET}

        # --- 2. 建立 Gurobi 模型 ---
        model = gp.Model("JIT_CF_AGV_Optimal_Routing")
        
        model.setParam('TimeLimit', 3600)  
        model.setParam('MIPGap', 0.0)     
        
        print(f"Gurobi 模型已建立，節點數: {len(N_SET)}，AGV數: {len(M_SET)}，任務數: {len(L_SET)}")

        # --- 3. 加入變數 (Variables) ---
        X = model.addVars(N_SET, N_SET, T_SET, M_SET, vtype=GRB.BINARY, name="X_move") 
        Y = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="Y_assign") 
        K = model.addVars(L_SET, T_SET, vtype=GRB.BINARY, name="K_not_pickup") 
        D = model.addVars(L_SET, T_SET, vtype=GRB.BINARY, name="D_not_delivery") 
        F = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0, name="F_deliv_penalty") 
        E = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0, name="E_pickup_penalty") 
        B = model.addVars(L_SET, T_SET, M_SET, vtype=GRB.BINARY, name="B_lin_pickup") 
        C = model.addVars(L_SET, T_SET, M_SET, vtype=GRB.BINARY, name="C_lin_delivery") 

        # --- 4. 設定目標函數 (論文 Eq. (2)) ---
        # 修正點：使用 T_INDEX 和 t (目標函數中的時間索引)
        model.setObjective(
            alpha * gp.quicksum(F[task] + E[task] for task in L_SET) + 
            beta * gp.quicksum(D[task, t] for task in L_SET for t in T_INDEX),
            GRB.MINIMIZE
        )

        # --- 5. 加入約束條件 (Constraints) ---

        # A. AGV 運動/流量約束
        # 修正點：所有外部迴圈和內部 quicksum 都使用 t
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


        # B. 衝突無限制約束 (論文 Eq. (6)-(8))
        # 修正點：使用 t
        model.addConstrs((gp.quicksum(X[j, i, t, agv] for agv in M_SET for j in A_i[i]) <= 1
                          for i in N_SET for t in T_SET), name="Constr6_NodeOccupancy")
        model.addConstrs((gp.quicksum(X[i, j, t, agv] + X[j, i, t, agv] for agv in M_SET) <= 1
                          for i in N_SET for j in A_i[i] if i < j for t in T_SET), name="Constr7_ArcConflict")
        model.addConstrs((gp.quicksum(X[j, i, t, m_i] for j in A_i[i]) + 
                          gp.quicksum(X[i, k, t, m_j] for k in A_i[i]) <= 1
                          for i in N_SET for t in T_SET for m_i in M_SET for m_j in M_SET if m_i != m_j), 
                          name="Constr8_InOutConflict")


        # C. 任務分配約束 (論文 Eq. (33))
        model.addConstrs((gp.quicksum(Y[task, agv] for agv in M_SET) == 1
                          for task in L_SET), name="Constr33_AssignOneVehicle")


        # D. 拾取/交付狀態變數 (K, D) 及其線性化輔助變數 (B, C)
        # 初始狀態 (論文 Eq. (17))
        model.addConstrs((K[task, 0] == 1 for task in L_SET), name="Constr17_K_initial")
        model.addConstrs((D[task, 0] == 1 for task in L_SET), name="Constr17_D_initial")

        for task in L_SET:
            u_l, g_l = Task_data[task]['u_l'], Task_data[task]['g_l']
            for t in T_SET[:-1]: # 修正點：使用 t
                
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

                # 狀態非增性 & 順序 (使用 t)
                model.addConstr(K[task, t] <= D[task, t], name=f"Constr18_PickupBeforeDelivery_{task}_{t}")
                model.addConstr(K[task, t+1] <= K[task, t], name=f"Constr18_K_NonIncrease_{task}_{t}")
                model.addConstr(D[task, t+1] <= D[task, t], name=f"Constr18_D_NonIncrease_{task}_{t}")


        # E. 懲罰定義約束 (論文 Eq. (19)-(20))
        for task in L_SET:
            
            # 使用 T_INDEX 和 t_idx (確保與外層迴圈 t 隔離)
            Pickup_Time = gp.quicksum(K[task, t_idx] for t_idx in T_INDEX) + 1
            Delivery_Time = gp.quicksum(D[task, t_idx] for t_idx in T_INDEX) + 1
            
            Desired_Pickup_Time = Task_data[task]['P_l^p']
            Desired_Delivery_Time = Task_data[task]['P_l^d']
            
            # (20) 拾取懲罰 e_l
            model.addConstr(E[task] >= Pickup_Time - Desired_Pickup_Time, name=f"Constr20_E_Tardiness_{task}")
            model.addConstr(E[task] >= Desired_Pickup_Time - Pickup_Time, name=f"Constr20_E_Earliness_{task}")
            
            # (19) 交付懲罰 f_l
            model.addConstr(F[task] >= Delivery_Time - Desired_Delivery_Time, name=f"Constr19_F_Tardiness_{task}")
            model.addConstr(F[task] >= Desired_Delivery_Time - Delivery_Time, name=f"Constr19_F_Earliness_{task}")

        # --- 6. 求解模型 (Solve Model) ---
        print("\n開始求解模型...")
        model.update()
        model.optimize()

        # --- 7. 輸出結果 (Analyze Results) ---
        print("\n==========================================================")
        if model.status == GRB.OPTIMAL or (model.status == GRB.TIME_LIMIT and model.SolCount > 0):
            print(f"目標函數值 (總完成時間): {model.objVal:.2f}")
        else:
            print("模型無可用解，無法列印 objVal。")
            
        # 輸出 AGV 分配結果
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            print("\n📌 任務分配 (Task Assignment):")
            for task in L_SET:
                for agv in M_SET:
                    if Y[task, agv].X > 0.5:
                        print(f"  - 任務 {task} (拾取點 {Task_data[task]['u_l']} -> 交付點 {Task_data[task]['g_l']}) 分配給 AGV {agv}")
    
            print("\n AGV 路由結果:")
            all_agv_routes = {}
            time_steps_output = list(T_SET)
            TOTAL_TIME_STEPS = len(T_SET)

            for agv in M_SET:
                route = []
                current_node = S_m[agv]
                for t in T_SET:
                    for i in N_SET:
                        if any(X[i, j, t, agv].X > 0.5 for j in A_i[i]):
                            current_node = i
                            break
                    route.append(current_node)
                all_agv_routes[agv] = np.asarray(route, dtype=float)
                print(f"  - AGV{agv} route length = {all_agv_routes[agv].shape[0]}")
                print(f"AGV{agv} = {route}")
            # 🔹 7️⃣ 繪圖（呼叫你的 plot_routes 函式）
            plot_routes(time_steps_output, all_agv_routes, TOTAL_TIME_STEPS)

        
    except gp.GurobiError as e:
        print(f"\n❌ Gurobi 錯誤代碼 {e.errno}: {e}")
        print("請確認 Gurobi 已安裝並取得有效授權。")
    except Exception as e:
        print(f"\n❌ 發生一般錯誤: {e}")