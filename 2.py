import gurobipy as gp
from gurobipy import GRB
import random

# --- 腳本開始：直接運行所有邏輯 ---
if __name__ == "__main__":
    try:
        # --- 1. 定義集合與參數 (Sets and Parameters) ---

        # 核心維度 (符合您的要求：3 AGV, 10 任務)
        M_SET = range(1, 4)           # 3 輛 AGV (AGV 索引 1, 2, 3)
        L_SET = range(1, 11)          # 10 個任務 
        N_SET = range(1, 16)          # 15 個節點 
        T_SET = range(30)             # 30 個時間步長 (t=0 到 t=29) 
        time = 1
        
        # 權重參數
        alpha = 1  # 早到/遲到懲罰權重 
        beta = 1   # 總完成時間權重 

        # 模擬 AGV 初始位置 
        S_m = {1: 1, 2: 5, 3: 10} 
        
        # 模擬任務資料
        Task_data = {}
        random.seed(42) 
        for task in L_SET:
            u_l = random.choice(N_SET)
            g_l = random.choice(N_SET)
            P_p = random.randint(5, 15)
            P_d = random.randint(P_p + 5, 25)
            Task_data[task] = {'u_l': u_l, 'g_l': g_l, 'P_l^p': P_p, 'P_l^d': P_d}

        # 模擬網絡結構: 鄰接節點 A_i 
        def get_adj_nodes(i):
            adj = {i}
            if i + 1 in N_SET: adj.add(i + 1)
            if i - 1 in N_SET: adj.add(i - 1)
            if i == 5: adj.add(10)
            if i == 10: adj.add(5)
            return list(adj)
        
        A_i = {i: get_adj_nodes(i) for i in N_SET}

        # --- 2. 建立 Gurobi 模型 (專為尋找最優解配置) ---
        model = gp.Model("JIT_CF_AGV_Optimal_Routing")
        
        # *** 尋找最優解的關鍵設定 ***
        model.setParam('TimeLimit', 300)  # 設置 300 秒 (5 分鐘) 求解時限
        model.setParam('MIPGap', 0.0)     # 要求找到證明最優的解 (Gap 必須為 0)
        
        print(f"Gurobi 模型已建立，目標：尋找最優解 (MIPGap=0.0，TimeLimit=300s)")

        # --- 3. 加入變數 (Variables) ---
        X = model.addVars(N_SET, N_SET, T_SET, M_SET, vtype=GRB.BINARY, name="X_move")
        Y = model.addVars(L_SET, M_SET, vtype=GRB.BINARY, name="Y_assign")
        K = model.addVars(L_SET, T_SET, vtype=GRB.BINARY, name="K_not_pickup")
        D = model.addVars(L_SET, T_SET, vtype=GRB.BINARY, name="D_not_delivery")
        F = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0, name="F_deliv_penalty")
        E = model.addVars(L_SET, vtype=GRB.CONTINUOUS, lb=0, name="E_pickup_penalty")
        B = model.addVars(L_SET, T_SET, M_SET, vtype=GRB.BINARY, name="B_lin_pickup")
        C = model.addVars(L_SET, T_SET, M_SET, vtype=GRB.BINARY, name="C_lin_delivery")

        # --- 4. 設定目標函數 (Objective Function) ---
        model.setObjective(
            alpha * gp.quicksum(F[task] + E[task] for task in L_SET) + 
            beta * gp.quicksum(D[task, time] for task in L_SET for time in T_SET),
            GRB.MINIMIZE
        )

        # --- 5. 加入約束條件 (Constraints) ---

        # A. AGV 運動/流量約束
        model.addConstrs((gp.quicksum(X[i, j, time, agv] for j in A_i[i]) <= 1
                          for i in N_SET for time in T_SET for agv in M_SET), name="Constr3_MaxOneMove")
        model.addConstrs((gp.quicksum(X[i, j, time, agv] for i in N_SET for j in A_i[i]) == 1
                          for time in T_SET for agv in M_SET), name="Constr4_MustMoveOrStop")
        model.addConstrs((gp.quicksum(X[j, i, time, agv] for j in A_i[i]) == 
                          gp.quicksum(X[i, k, time+1, agv] for k in A_i[i])
                          for i in N_SET for time in T_SET[:-1] for agv in M_SET), name="Constr5_FlowConservation")
        
        # (9)-(10) 初始位置約束 (t=0)
        for agv in M_SET:
            start_node = S_m[agv]
            model.addConstr(gp.quicksum(X[start_node, j, 0, agv] for j in A_i[start_node]) == 1, name=f"Constr9_StartNode_{agv}")
            model.addConstrs((gp.quicksum(X[i, j, 0, agv] for j in A_i[i]) == 0
                              for i in N_SET if i != start_node), name=f"Constr10_OtherNodes_{agv}")


        # B. 衝突無限制約束
        model.addConstrs((gp.quicksum(X[j, i, time, agv] for agv in M_SET for j in A_i[i]) <= 1
                          for i in N_SET for time in T_SET), name="Constr6_NodeOccupancy") 
        model.addConstrs((gp.quicksum(X[i, j, time, agv] + X[j, i, time, agv] for agv in M_SET) <= 1
                          for i in N_SET for j in A_i[i] if i < j), name="Constr7_ArcConflict") 
        model.addConstrs((gp.quicksum(X[j, i, time, m_i] for j in A_i[i]) + 
                          gp.quicksum(X[i, k, time, m_j] for k in A_i[i]) <= 1
                          for i in N_SET for time in T_SET for m_i in M_SET for m_j in M_SET if m_i != m_j), 
                          name="Constr8_InOutConflict") 


        # C. 任務分配約束
        model.addConstrs((gp.quicksum(Y[task, agv] for agv in M_SET) == 1
                          for task in L_SET), name="Constr33_AssignOneVehicle") 


        # D. 拾取/交付狀態變數 (K, D) 及其線性化輔助變數 (B, C)
        model.addConstrs((K[task, 0] == 1 for task in L_SET), name="Constr17_K_initial") 
        model.addConstrs((D[task, 0] == 1 for task in L_SET), name="Constr17_D_initial") 

        for task in L_SET:
            u_l, g_l = Task_data[task]['u_l'], Task_data[task]['g_l']
            for time in T_SET[:-1]:
                
                # --- 線性化約束 (處理乘積 Y*X) --- 
                for agv in M_SET:
                    model.addConstr(B[task, time, agv] <= Y[task, agv], name=f"Lin_B1_{task}_{time}_{agv}")
                    model.addConstr(B[task, time, agv] <= X[u_l, u_l, time, agv], name=f"Lin_B2_{task}_{time}_{agv}")
                    model.addConstr(B[task, time, agv] >= Y[task, agv] + X[u_l, u_l, time, agv] - 1, name=f"Lin_B3_{task}_{time}_{agv}")
                    model.addConstr(C[task, time, agv] <= Y[task, agv], name=f"Lin_C1_{task}_{time}_{agv}")
                    model.addConstr(C[task, time, agv] <= X[g_l, g_l, time, agv], name=f"Lin_C2_{task}_{time}_{agv}")
                    model.addConstr(C[task, time, agv] >= Y[task, agv] + X[g_l, g_l, time, agv] - 1, name=f"Lin_C3_{task}_{time}_{agv}")

                # --- 狀態轉換約束 --- 
                model.addConstr(K[task, time] - gp.quicksum(B[task, time, agv] for agv in M_SET) <= K[task, time+1], name=f"Constr13_K_Update1_{task}_{time}")
                model.addConstr(gp.quicksum(B[task, time, agv] for agv in M_SET) + K[task, time+1] <= 1, name=f"Constr14_K_Update2_{task}_{time}")
                model.addConstr(D[task, time] - gp.quicksum(C[task, time, agv] for agv in M_SET) <= D[task, time+1], name=f"Constr15_D_Update1_{task}_{time}")
                model.addConstr(gp.quicksum(C[task, time, agv] for agv in M_SET) + D[task, time+1] <= 1, name=f"Constr16_D_Update2_{task}_{time}")

                # (18) 狀態非增性 & 順序 (K <= D)
                model.addConstr(K[task, time] <= D[task, time], name=f"Constr18_PickupBeforeDelivery_{task}_{time}")
                model.addConstr(K[task, time+1] <= K[task, time], name=f"Constr18_K_NonIncrease_{task}_{time}")
                model.addConstr(D[task, time+1] <= D[task, time], name=f"Constr18_D_NonIncrease_{task}_{time}")


        # E. 懲罰定義約束
        for task in L_SET:
            
            # 使用 t_idx 替代 time
            time_index = T_SET 
            Pickup_Time = gp.quicksum(K[task, t_idx] for t_idx in time_index) + 1
            Delivery_Time = gp.quicksum(D[task, t_idx] for t_idx in time_index) + 1
            
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
        if model.status == GRB.OPTIMAL:
            print(f"🎉 模型找到最優解！")
        elif model.status == GRB.TIME_LIMIT:
            print(f"⏱️ 達到 {model.Params.TimeLimit} 秒時間限制。找到的最佳可行解如下。")
            print(f"   (最優性可能未證實，MIPGap 為 {model.MIPGap * 100:.4f}%)")
        else:
             print(f"❌ 模型求解失敗或無可行解。Gurobi 狀態碼: {model.status}")
             

        print(f"目標函數值 (總懲罰/完成時間): {model.objVal:.2f}")
        print("==========================================================")
            
        # 輸出 AGV 分配結果
        print("\n📌 任務分配 (Task Assignment):")
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            for task in L_SET:
                for agv in M_SET:
                    if Y[task, agv].X > 0.5:
                        print(f"  - 任務 {task} (拾取點 {Task_data[task]['u_l']} -> 交付點 {Task_data[task]['g_l']}) 分配給 AGV {agv}")

        # 輸出 AGV 路由結果 (節點 X 時間) - 僅顯示前 10 個時間步
        print("\n📌 AGV 路由結果 (前 10 個時間步):")
        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            for agv in M_SET:
                route = []
                current_node = S_m[agv]
                for time in T_SET[:10]:
                    for i in N_SET:
                        if any(X[i, j, time, agv].X > 0.5 for j in A_i[i]):
                            current_node = i
                            break
                    route.append(current_node)
                print(f"  - AGV {agv} 路由 (t=0到9): {route}")
        
    except gp.GurobiError as e:
        print(f"\n❌ Gurobi 錯誤代碼 {e.errno}: {e}")
        print("請確認 Gurobi 已安裝並取得有效授權。")
    except Exception as e:
        print(f"\n❌ 發生一般錯誤: {e}")