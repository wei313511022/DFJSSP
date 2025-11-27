\ Model Reschedule_event
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0x4aa3fc210d6b739c
Minimize
  0 W_seq[0,0,1] + 0 W_seq[0,0,2] + 0 W_seq[0,0,3] + 0 W_seq[0,6,1]
   + 0 W_seq[0,6,2] + 0 W_seq[0,6,3] + 0 W_seq[1,1,1] + 0 W_seq[1,1,2]
   + 0 W_seq[1,1,3] + 0 W_seq[2,2,1] + 0 W_seq[2,2,2] + 0 W_seq[2,2,3]
   + 0 W_seq[3,3,1] + 0 W_seq[3,3,2] + 0 W_seq[3,3,3] + 0 W_seq[4,4,1]
   + 0 W_seq[4,4,2] + 0 W_seq[4,4,3] + 0 W_seq[5,5,1] + 0 W_seq[5,5,2]
   + 0 W_seq[5,5,3] + 0 W_seq[6,0,1] + 0 W_seq[6,0,2] + 0 W_seq[6,0,3]
   + 0 W_seq[6,6,1] + 0 W_seq[6,6,2] + 0 W_seq[6,6,3] + T_end[1] + T_end[2]
   + T_end[3] + T_end[4] + T_end[5]
Subject To
 C1_one_pick[1]: A_pick[1,1] + A_pick[1,10] + A_pick[1,6] = 1
 C1_one_pick[2]: A_pick[2,1] + A_pick[2,10] + A_pick[2,6] = 1
 C1_one_pick[3]: A_pick[3,1] + A_pick[3,10] + A_pick[3,6] = 1
 C1_one_pick[4]: A_pick[4,1] + A_pick[4,10] + A_pick[4,6] = 1
 C1_one_pick[5]: A_pick[5,1] + A_pick[5,10] + A_pick[5,6] = 1
 C2_assign[1]: Y_assign[1,1] + Y_assign[1,2] + Y_assign[1,3] = 1
 C2_assign[2]: Y_assign[2,1] + Y_assign[2,2] + Y_assign[2,3] = 1
 C2_assign[3]: Y_assign[3,1] + Y_assign[3,2] + Y_assign[3,3] = 1
 C2_assign[4]: Y_assign[4,1] + Y_assign[4,2] + Y_assign[4,3] = 1
 C2_assign[5]: Y_assign[5,1] + Y_assign[5,2] + Y_assign[5,3] = 1
 C3_pred_1: W_seq[0,1,1] + W_seq[0,1,2] + W_seq[0,1,3] + W_seq[2,1,1]
   + W_seq[2,1,2] + W_seq[2,1,3] + W_seq[3,1,1] + W_seq[3,1,2]
   + W_seq[3,1,3] + W_seq[4,1,1] + W_seq[4,1,2] + W_seq[4,1,3]
   + W_seq[5,1,1] + W_seq[5,1,2] + W_seq[5,1,3] + W_seq[6,1,1]
   + W_seq[6,1,2] + W_seq[6,1,3] = 1
 C3_succ_1: W_seq[1,0,1] + W_seq[1,0,2] + W_seq[1,0,3] + W_seq[1,2,1]
   + W_seq[1,2,2] + W_seq[1,2,3] + W_seq[1,3,1] + W_seq[1,3,2]
   + W_seq[1,3,3] + W_seq[1,4,1] + W_seq[1,4,2] + W_seq[1,4,3]
   + W_seq[1,5,1] + W_seq[1,5,2] + W_seq[1,5,3] + W_seq[1,6,1]
   + W_seq[1,6,2] + W_seq[1,6,3] = 1
 C3_link_in_1[0,1]: - Y_assign[1,1] + W_seq[0,1,1] <= 0
 C3_link_in_1[0,2]: - Y_assign[1,2] + W_seq[0,1,2] <= 0
 C3_link_in_1[0,3]: - Y_assign[1,3] + W_seq[0,1,3] <= 0
 C3_link_in_1[2,1]: - Y_assign[1,1] + W_seq[2,1,1] <= 0
 C3_link_in_1[2,2]: - Y_assign[1,2] + W_seq[2,1,2] <= 0
 C3_link_in_1[2,3]: - Y_assign[1,3] + W_seq[2,1,3] <= 0
 C3_link_in_1[3,1]: - Y_assign[1,1] + W_seq[3,1,1] <= 0
 C3_link_in_1[3,2]: - Y_assign[1,2] + W_seq[3,1,2] <= 0
 C3_link_in_1[3,3]: - Y_assign[1,3] + W_seq[3,1,3] <= 0
 C3_link_in_1[4,1]: - Y_assign[1,1] + W_seq[4,1,1] <= 0
 C3_link_in_1[4,2]: - Y_assign[1,2] + W_seq[4,1,2] <= 0
 C3_link_in_1[4,3]: - Y_assign[1,3] + W_seq[4,1,3] <= 0
 C3_link_in_1[5,1]: - Y_assign[1,1] + W_seq[5,1,1] <= 0
 C3_link_in_1[5,2]: - Y_assign[1,2] + W_seq[5,1,2] <= 0
 C3_link_in_1[5,3]: - Y_assign[1,3] + W_seq[5,1,3] <= 0
 C3_link_in_1[6,1]: - Y_assign[1,1] + W_seq[6,1,1] <= 0
 C3_link_in_1[6,2]: - Y_assign[1,2] + W_seq[6,1,2] <= 0
 C3_link_in_1[6,3]: - Y_assign[1,3] + W_seq[6,1,3] <= 0
 C3_link_out_1[0,1]: - Y_assign[1,1] + W_seq[1,0,1] <= 0
 C3_link_out_1[0,2]: - Y_assign[1,2] + W_seq[1,0,2] <= 0
 C3_link_out_1[0,3]: - Y_assign[1,3] + W_seq[1,0,3] <= 0
 C3_link_out_1[2,1]: - Y_assign[1,1] + W_seq[1,2,1] <= 0
 C3_link_out_1[2,2]: - Y_assign[1,2] + W_seq[1,2,2] <= 0
 C3_link_out_1[2,3]: - Y_assign[1,3] + W_seq[1,2,3] <= 0
 C3_link_out_1[3,1]: - Y_assign[1,1] + W_seq[1,3,1] <= 0
 C3_link_out_1[3,2]: - Y_assign[1,2] + W_seq[1,3,2] <= 0
 C3_link_out_1[3,3]: - Y_assign[1,3] + W_seq[1,3,3] <= 0
 C3_link_out_1[4,1]: - Y_assign[1,1] + W_seq[1,4,1] <= 0
 C3_link_out_1[4,2]: - Y_assign[1,2] + W_seq[1,4,2] <= 0
 C3_link_out_1[4,3]: - Y_assign[1,3] + W_seq[1,4,3] <= 0
 C3_link_out_1[5,1]: - Y_assign[1,1] + W_seq[1,5,1] <= 0
 C3_link_out_1[5,2]: - Y_assign[1,2] + W_seq[1,5,2] <= 0
 C3_link_out_1[5,3]: - Y_assign[1,3] + W_seq[1,5,3] <= 0
 C3_link_out_1[6,1]: - Y_assign[1,1] + W_seq[1,6,1] <= 0
 C3_link_out_1[6,2]: - Y_assign[1,2] + W_seq[1,6,2] <= 0
 C3_link_out_1[6,3]: - Y_assign[1,3] + W_seq[1,6,3] <= 0
 C3_pred_2: W_seq[0,2,1] + W_seq[0,2,2] + W_seq[0,2,3] + W_seq[1,2,1]
   + W_seq[1,2,2] + W_seq[1,2,3] + W_seq[3,2,1] + W_seq[3,2,2]
   + W_seq[3,2,3] + W_seq[4,2,1] + W_seq[4,2,2] + W_seq[4,2,3]
   + W_seq[5,2,1] + W_seq[5,2,2] + W_seq[5,2,3] + W_seq[6,2,1]
   + W_seq[6,2,2] + W_seq[6,2,3] = 1
 C3_succ_2: W_seq[2,0,1] + W_seq[2,0,2] + W_seq[2,0,3] + W_seq[2,1,1]
   + W_seq[2,1,2] + W_seq[2,1,3] + W_seq[2,3,1] + W_seq[2,3,2]
   + W_seq[2,3,3] + W_seq[2,4,1] + W_seq[2,4,2] + W_seq[2,4,3]
   + W_seq[2,5,1] + W_seq[2,5,2] + W_seq[2,5,3] + W_seq[2,6,1]
   + W_seq[2,6,2] + W_seq[2,6,3] = 1
 C3_link_in_2[0,1]: - Y_assign[2,1] + W_seq[0,2,1] <= 0
 C3_link_in_2[0,2]: - Y_assign[2,2] + W_seq[0,2,2] <= 0
 C3_link_in_2[0,3]: - Y_assign[2,3] + W_seq[0,2,3] <= 0
 C3_link_in_2[1,1]: - Y_assign[2,1] + W_seq[1,2,1] <= 0
 C3_link_in_2[1,2]: - Y_assign[2,2] + W_seq[1,2,2] <= 0
 C3_link_in_2[1,3]: - Y_assign[2,3] + W_seq[1,2,3] <= 0
 C3_link_in_2[3,1]: - Y_assign[2,1] + W_seq[3,2,1] <= 0
 C3_link_in_2[3,2]: - Y_assign[2,2] + W_seq[3,2,2] <= 0
 C3_link_in_2[3,3]: - Y_assign[2,3] + W_seq[3,2,3] <= 0
 C3_link_in_2[4,1]: - Y_assign[2,1] + W_seq[4,2,1] <= 0
 C3_link_in_2[4,2]: - Y_assign[2,2] + W_seq[4,2,2] <= 0
 C3_link_in_2[4,3]: - Y_assign[2,3] + W_seq[4,2,3] <= 0
 C3_link_in_2[5,1]: - Y_assign[2,1] + W_seq[5,2,1] <= 0
 C3_link_in_2[5,2]: - Y_assign[2,2] + W_seq[5,2,2] <= 0
 C3_link_in_2[5,3]: - Y_assign[2,3] + W_seq[5,2,3] <= 0
 C3_link_in_2[6,1]: - Y_assign[2,1] + W_seq[6,2,1] <= 0
 C3_link_in_2[6,2]: - Y_assign[2,2] + W_seq[6,2,2] <= 0
 C3_link_in_2[6,3]: - Y_assign[2,3] + W_seq[6,2,3] <= 0
 C3_link_out_2[0,1]: - Y_assign[2,1] + W_seq[2,0,1] <= 0
 C3_link_out_2[0,2]: - Y_assign[2,2] + W_seq[2,0,2] <= 0
 C3_link_out_2[0,3]: - Y_assign[2,3] + W_seq[2,0,3] <= 0
 C3_link_out_2[1,1]: - Y_assign[2,1] + W_seq[2,1,1] <= 0
 C3_link_out_2[1,2]: - Y_assign[2,2] + W_seq[2,1,2] <= 0
 C3_link_out_2[1,3]: - Y_assign[2,3] + W_seq[2,1,3] <= 0
 C3_link_out_2[3,1]: - Y_assign[2,1] + W_seq[2,3,1] <= 0
 C3_link_out_2[3,2]: - Y_assign[2,2] + W_seq[2,3,2] <= 0
 C3_link_out_2[3,3]: - Y_assign[2,3] + W_seq[2,3,3] <= 0
 C3_link_out_2[4,1]: - Y_assign[2,1] + W_seq[2,4,1] <= 0
 C3_link_out_2[4,2]: - Y_assign[2,2] + W_seq[2,4,2] <= 0
 C3_link_out_2[4,3]: - Y_assign[2,3] + W_seq[2,4,3] <= 0
 C3_link_out_2[5,1]: - Y_assign[2,1] + W_seq[2,5,1] <= 0
 C3_link_out_2[5,2]: - Y_assign[2,2] + W_seq[2,5,2] <= 0
 C3_link_out_2[5,3]: - Y_assign[2,3] + W_seq[2,5,3] <= 0
 C3_link_out_2[6,1]: - Y_assign[2,1] + W_seq[2,6,1] <= 0
 C3_link_out_2[6,2]: - Y_assign[2,2] + W_seq[2,6,2] <= 0
 C3_link_out_2[6,3]: - Y_assign[2,3] + W_seq[2,6,3] <= 0
 C3_pred_3: W_seq[0,3,1] + W_seq[0,3,2] + W_seq[0,3,3] + W_seq[1,3,1]
   + W_seq[1,3,2] + W_seq[1,3,3] + W_seq[2,3,1] + W_seq[2,3,2]
   + W_seq[2,3,3] + W_seq[4,3,1] + W_seq[4,3,2] + W_seq[4,3,3]
   + W_seq[5,3,1] + W_seq[5,3,2] + W_seq[5,3,3] + W_seq[6,3,1]
   + W_seq[6,3,2] + W_seq[6,3,3] = 1
 C3_succ_3: W_seq[3,0,1] + W_seq[3,0,2] + W_seq[3,0,3] + W_seq[3,1,1]
   + W_seq[3,1,2] + W_seq[3,1,3] + W_seq[3,2,1] + W_seq[3,2,2]
   + W_seq[3,2,3] + W_seq[3,4,1] + W_seq[3,4,2] + W_seq[3,4,3]
   + W_seq[3,5,1] + W_seq[3,5,2] + W_seq[3,5,3] + W_seq[3,6,1]
   + W_seq[3,6,2] + W_seq[3,6,3] = 1
 C3_link_in_3[0,1]: - Y_assign[3,1] + W_seq[0,3,1] <= 0
 C3_link_in_3[0,2]: - Y_assign[3,2] + W_seq[0,3,2] <= 0
 C3_link_in_3[0,3]: - Y_assign[3,3] + W_seq[0,3,3] <= 0
 C3_link_in_3[1,1]: - Y_assign[3,1] + W_seq[1,3,1] <= 0
 C3_link_in_3[1,2]: - Y_assign[3,2] + W_seq[1,3,2] <= 0
 C3_link_in_3[1,3]: - Y_assign[3,3] + W_seq[1,3,3] <= 0
 C3_link_in_3[2,1]: - Y_assign[3,1] + W_seq[2,3,1] <= 0
 C3_link_in_3[2,2]: - Y_assign[3,2] + W_seq[2,3,2] <= 0
 C3_link_in_3[2,3]: - Y_assign[3,3] + W_seq[2,3,3] <= 0
 C3_link_in_3[4,1]: - Y_assign[3,1] + W_seq[4,3,1] <= 0
 C3_link_in_3[4,2]: - Y_assign[3,2] + W_seq[4,3,2] <= 0
 C3_link_in_3[4,3]: - Y_assign[3,3] + W_seq[4,3,3] <= 0
 C3_link_in_3[5,1]: - Y_assign[3,1] + W_seq[5,3,1] <= 0
 C3_link_in_3[5,2]: - Y_assign[3,2] + W_seq[5,3,2] <= 0
 C3_link_in_3[5,3]: - Y_assign[3,3] + W_seq[5,3,3] <= 0
 C3_link_in_3[6,1]: - Y_assign[3,1] + W_seq[6,3,1] <= 0
 C3_link_in_3[6,2]: - Y_assign[3,2] + W_seq[6,3,2] <= 0
 C3_link_in_3[6,3]: - Y_assign[3,3] + W_seq[6,3,3] <= 0
 C3_link_out_3[0,1]: - Y_assign[3,1] + W_seq[3,0,1] <= 0
 C3_link_out_3[0,2]: - Y_assign[3,2] + W_seq[3,0,2] <= 0
 C3_link_out_3[0,3]: - Y_assign[3,3] + W_seq[3,0,3] <= 0
 C3_link_out_3[1,1]: - Y_assign[3,1] + W_seq[3,1,1] <= 0
 C3_link_out_3[1,2]: - Y_assign[3,2] + W_seq[3,1,2] <= 0
 C3_link_out_3[1,3]: - Y_assign[3,3] + W_seq[3,1,3] <= 0
 C3_link_out_3[2,1]: - Y_assign[3,1] + W_seq[3,2,1] <= 0
 C3_link_out_3[2,2]: - Y_assign[3,2] + W_seq[3,2,2] <= 0
 C3_link_out_3[2,3]: - Y_assign[3,3] + W_seq[3,2,3] <= 0
 C3_link_out_3[4,1]: - Y_assign[3,1] + W_seq[3,4,1] <= 0
 C3_link_out_3[4,2]: - Y_assign[3,2] + W_seq[3,4,2] <= 0
 C3_link_out_3[4,3]: - Y_assign[3,3] + W_seq[3,4,3] <= 0
 C3_link_out_3[5,1]: - Y_assign[3,1] + W_seq[3,5,1] <= 0
 C3_link_out_3[5,2]: - Y_assign[3,2] + W_seq[3,5,2] <= 0
 C3_link_out_3[5,3]: - Y_assign[3,3] + W_seq[3,5,3] <= 0
 C3_link_out_3[6,1]: - Y_assign[3,1] + W_seq[3,6,1] <= 0
 C3_link_out_3[6,2]: - Y_assign[3,2] + W_seq[3,6,2] <= 0
 C3_link_out_3[6,3]: - Y_assign[3,3] + W_seq[3,6,3] <= 0
 C3_pred_4: W_seq[0,4,1] + W_seq[0,4,2] + W_seq[0,4,3] + W_seq[1,4,1]
   + W_seq[1,4,2] + W_seq[1,4,3] + W_seq[2,4,1] + W_seq[2,4,2]
   + W_seq[2,4,3] + W_seq[3,4,1] + W_seq[3,4,2] + W_seq[3,4,3]
   + W_seq[5,4,1] + W_seq[5,4,2] + W_seq[5,4,3] + W_seq[6,4,1]
   + W_seq[6,4,2] + W_seq[6,4,3] = 1
 C3_succ_4: W_seq[4,0,1] + W_seq[4,0,2] + W_seq[4,0,3] + W_seq[4,1,1]
   + W_seq[4,1,2] + W_seq[4,1,3] + W_seq[4,2,1] + W_seq[4,2,2]
   + W_seq[4,2,3] + W_seq[4,3,1] + W_seq[4,3,2] + W_seq[4,3,3]
   + W_seq[4,5,1] + W_seq[4,5,2] + W_seq[4,5,3] + W_seq[4,6,1]
   + W_seq[4,6,2] + W_seq[4,6,3] = 1
 C3_link_in_4[0,1]: - Y_assign[4,1] + W_seq[0,4,1] <= 0
 C3_link_in_4[0,2]: - Y_assign[4,2] + W_seq[0,4,2] <= 0
 C3_link_in_4[0,3]: - Y_assign[4,3] + W_seq[0,4,3] <= 0
 C3_link_in_4[1,1]: - Y_assign[4,1] + W_seq[1,4,1] <= 0
 C3_link_in_4[1,2]: - Y_assign[4,2] + W_seq[1,4,2] <= 0
 C3_link_in_4[1,3]: - Y_assign[4,3] + W_seq[1,4,3] <= 0
 C3_link_in_4[2,1]: - Y_assign[4,1] + W_seq[2,4,1] <= 0
 C3_link_in_4[2,2]: - Y_assign[4,2] + W_seq[2,4,2] <= 0
 C3_link_in_4[2,3]: - Y_assign[4,3] + W_seq[2,4,3] <= 0
 C3_link_in_4[3,1]: - Y_assign[4,1] + W_seq[3,4,1] <= 0
 C3_link_in_4[3,2]: - Y_assign[4,2] + W_seq[3,4,2] <= 0
 C3_link_in_4[3,3]: - Y_assign[4,3] + W_seq[3,4,3] <= 0
 C3_link_in_4[5,1]: - Y_assign[4,1] + W_seq[5,4,1] <= 0
 C3_link_in_4[5,2]: - Y_assign[4,2] + W_seq[5,4,2] <= 0
 C3_link_in_4[5,3]: - Y_assign[4,3] + W_seq[5,4,3] <= 0
 C3_link_in_4[6,1]: - Y_assign[4,1] + W_seq[6,4,1] <= 0
 C3_link_in_4[6,2]: - Y_assign[4,2] + W_seq[6,4,2] <= 0
 C3_link_in_4[6,3]: - Y_assign[4,3] + W_seq[6,4,3] <= 0
 C3_link_out_4[0,1]: - Y_assign[4,1] + W_seq[4,0,1] <= 0
 C3_link_out_4[0,2]: - Y_assign[4,2] + W_seq[4,0,2] <= 0
 C3_link_out_4[0,3]: - Y_assign[4,3] + W_seq[4,0,3] <= 0
 C3_link_out_4[1,1]: - Y_assign[4,1] + W_seq[4,1,1] <= 0
 C3_link_out_4[1,2]: - Y_assign[4,2] + W_seq[4,1,2] <= 0
 C3_link_out_4[1,3]: - Y_assign[4,3] + W_seq[4,1,3] <= 0
 C3_link_out_4[2,1]: - Y_assign[4,1] + W_seq[4,2,1] <= 0
 C3_link_out_4[2,2]: - Y_assign[4,2] + W_seq[4,2,2] <= 0
 C3_link_out_4[2,3]: - Y_assign[4,3] + W_seq[4,2,3] <= 0
 C3_link_out_4[3,1]: - Y_assign[4,1] + W_seq[4,3,1] <= 0
 C3_link_out_4[3,2]: - Y_assign[4,2] + W_seq[4,3,2] <= 0
 C3_link_out_4[3,3]: - Y_assign[4,3] + W_seq[4,3,3] <= 0
 C3_link_out_4[5,1]: - Y_assign[4,1] + W_seq[4,5,1] <= 0
 C3_link_out_4[5,2]: - Y_assign[4,2] + W_seq[4,5,2] <= 0
 C3_link_out_4[5,3]: - Y_assign[4,3] + W_seq[4,5,3] <= 0
 C3_link_out_4[6,1]: - Y_assign[4,1] + W_seq[4,6,1] <= 0
 C3_link_out_4[6,2]: - Y_assign[4,2] + W_seq[4,6,2] <= 0
 C3_link_out_4[6,3]: - Y_assign[4,3] + W_seq[4,6,3] <= 0
 C3_pred_5: W_seq[0,5,1] + W_seq[0,5,2] + W_seq[0,5,3] + W_seq[1,5,1]
   + W_seq[1,5,2] + W_seq[1,5,3] + W_seq[2,5,1] + W_seq[2,5,2]
   + W_seq[2,5,3] + W_seq[3,5,1] + W_seq[3,5,2] + W_seq[3,5,3]
   + W_seq[4,5,1] + W_seq[4,5,2] + W_seq[4,5,3] + W_seq[6,5,1]
   + W_seq[6,5,2] + W_seq[6,5,3] = 1
 C3_succ_5: W_seq[5,0,1] + W_seq[5,0,2] + W_seq[5,0,3] + W_seq[5,1,1]
   + W_seq[5,1,2] + W_seq[5,1,3] + W_seq[5,2,1] + W_seq[5,2,2]
   + W_seq[5,2,3] + W_seq[5,3,1] + W_seq[5,3,2] + W_seq[5,3,3]
   + W_seq[5,4,1] + W_seq[5,4,2] + W_seq[5,4,3] + W_seq[5,6,1]
   + W_seq[5,6,2] + W_seq[5,6,3] = 1
 C3_link_in_5[0,1]: - Y_assign[5,1] + W_seq[0,5,1] <= 0
 C3_link_in_5[0,2]: - Y_assign[5,2] + W_seq[0,5,2] <= 0
 C3_link_in_5[0,3]: - Y_assign[5,3] + W_seq[0,5,3] <= 0
 C3_link_in_5[1,1]: - Y_assign[5,1] + W_seq[1,5,1] <= 0
 C3_link_in_5[1,2]: - Y_assign[5,2] + W_seq[1,5,2] <= 0
 C3_link_in_5[1,3]: - Y_assign[5,3] + W_seq[1,5,3] <= 0
 C3_link_in_5[2,1]: - Y_assign[5,1] + W_seq[2,5,1] <= 0
 C3_link_in_5[2,2]: - Y_assign[5,2] + W_seq[2,5,2] <= 0
 C3_link_in_5[2,3]: - Y_assign[5,3] + W_seq[2,5,3] <= 0
 C3_link_in_5[3,1]: - Y_assign[5,1] + W_seq[3,5,1] <= 0
 C3_link_in_5[3,2]: - Y_assign[5,2] + W_seq[3,5,2] <= 0
 C3_link_in_5[3,3]: - Y_assign[5,3] + W_seq[3,5,3] <= 0
 C3_link_in_5[4,1]: - Y_assign[5,1] + W_seq[4,5,1] <= 0
 C3_link_in_5[4,2]: - Y_assign[5,2] + W_seq[4,5,2] <= 0
 C3_link_in_5[4,3]: - Y_assign[5,3] + W_seq[4,5,3] <= 0
 C3_link_in_5[6,1]: - Y_assign[5,1] + W_seq[6,5,1] <= 0
 C3_link_in_5[6,2]: - Y_assign[5,2] + W_seq[6,5,2] <= 0
 C3_link_in_5[6,3]: - Y_assign[5,3] + W_seq[6,5,3] <= 0
 C3_link_out_5[0,1]: - Y_assign[5,1] + W_seq[5,0,1] <= 0
 C3_link_out_5[0,2]: - Y_assign[5,2] + W_seq[5,0,2] <= 0
 C3_link_out_5[0,3]: - Y_assign[5,3] + W_seq[5,0,3] <= 0
 C3_link_out_5[1,1]: - Y_assign[5,1] + W_seq[5,1,1] <= 0
 C3_link_out_5[1,2]: - Y_assign[5,2] + W_seq[5,1,2] <= 0
 C3_link_out_5[1,3]: - Y_assign[5,3] + W_seq[5,1,3] <= 0
 C3_link_out_5[2,1]: - Y_assign[5,1] + W_seq[5,2,1] <= 0
 C3_link_out_5[2,2]: - Y_assign[5,2] + W_seq[5,2,2] <= 0
 C3_link_out_5[2,3]: - Y_assign[5,3] + W_seq[5,2,3] <= 0
 C3_link_out_5[3,1]: - Y_assign[5,1] + W_seq[5,3,1] <= 0
 C3_link_out_5[3,2]: - Y_assign[5,2] + W_seq[5,3,2] <= 0
 C3_link_out_5[3,3]: - Y_assign[5,3] + W_seq[5,3,3] <= 0
 C3_link_out_5[4,1]: - Y_assign[5,1] + W_seq[5,4,1] <= 0
 C3_link_out_5[4,2]: - Y_assign[5,2] + W_seq[5,4,2] <= 0
 C3_link_out_5[4,3]: - Y_assign[5,3] + W_seq[5,4,3] <= 0
 C3_link_out_5[6,1]: - Y_assign[5,1] + W_seq[5,6,1] <= 0
 C3_link_out_5[6,2]: - Y_assign[5,2] + W_seq[5,6,2] <= 0
 C3_link_out_5[6,3]: - Y_assign[5,3] + W_seq[5,6,3] <= 0
 C4_start_1: W_seq[0,1,1] + W_seq[0,2,1] + W_seq[0,3,1] + W_seq[0,4,1]
   + W_seq[0,5,1] = 1
 C4_end_1: W_seq[1,6,1] + W_seq[2,6,1] + W_seq[3,6,1] + W_seq[4,6,1]
   + W_seq[5,6,1] = 1
 C4_start_2: W_seq[0,1,2] + W_seq[0,2,2] + W_seq[0,3,2] + W_seq[0,4,2]
   + W_seq[0,5,2] = 1
 C4_end_2: W_seq[1,6,2] + W_seq[2,6,2] + W_seq[3,6,2] + W_seq[4,6,2]
   + W_seq[5,6,2] = 1
 C4_start_3: W_seq[0,1,3] + W_seq[0,2,3] + W_seq[0,3,3] + W_seq[0,4,3]
   + W_seq[0,5,3] = 1
 C4_end_3: W_seq[1,6,3] + W_seq[2,6,3] + W_seq[3,6,3] + W_seq[4,6,3]
   + W_seq[5,6,3] = 1
 C5_no_return_start[1,1]: W_seq[1,0,1] = 0
 C5_no_return_start[1,2]: W_seq[1,0,2] = 0
 C5_no_return_start[1,3]: W_seq[1,0,3] = 0
 C5_no_return_start[2,1]: W_seq[2,0,1] = 0
 C5_no_return_start[2,2]: W_seq[2,0,2] = 0
 C5_no_return_start[2,3]: W_seq[2,0,3] = 0
 C5_no_return_start[3,1]: W_seq[3,0,1] = 0
 C5_no_return_start[3,2]: W_seq[3,0,2] = 0
 C5_no_return_start[3,3]: W_seq[3,0,3] = 0
 C5_no_return_start[4,1]: W_seq[4,0,1] = 0
 C5_no_return_start[4,2]: W_seq[4,0,2] = 0
 C5_no_return_start[4,3]: W_seq[4,0,3] = 0
 C5_no_return_start[5,1]: W_seq[5,0,1] = 0
 C5_no_return_start[5,2]: W_seq[5,0,2] = 0
 C5_no_return_start[5,3]: W_seq[5,0,3] = 0
 C6_exec[1]: - T_del[1] + T_end[1] = 15
 C6_exec[2]: - T_del[2] + T_end[2] = 10
 C6_exec[3]: - T_del[3] + T_end[3] = 10
 C6_exec[4]: - T_del[4] + T_end[4] = 20
 C6_exec[5]: - T_del[5] + T_end[5] = 10
 C_arrival[1]: T_pick[1] >= 0
 C_arrival[2]: T_pick[2] >= 0
 C_arrival[3]: T_pick[3] >= 0
 C_arrival[4]: T_pick[4] >= 0
 C_arrival[5]: T_pick[5] >= 0
 C7_trans_1_1: - 100 A_pick[1,1] - T_pick[1] + T_del[1] >= -84
 C7_trans_1_10: - 100 A_pick[1,10] - T_pick[1] + T_del[1] >= -89
 C7_trans_1_6: - 100 A_pick[1,6] - T_pick[1] + T_del[1] >= -89
 C7_trans_2_1: - 100 A_pick[2,1] - T_pick[2] + T_del[2] >= -89
 C7_trans_2_10: - 100 A_pick[2,10] - T_pick[2] + T_del[2] >= -84
 C7_trans_2_6: - 100 A_pick[2,6] - T_pick[2] + T_del[2] >= -88
 C7_trans_3_1: - 100 A_pick[3,1] - T_pick[3] + T_del[3] >= -89
 C7_trans_3_10: - 100 A_pick[3,10] - T_pick[3] + T_del[3] >= -84
 C7_trans_3_6: - 100 A_pick[3,6] - T_pick[3] + T_del[3] >= -88
 C7_trans_4_1: - 100 A_pick[4,1] - T_pick[4] + T_del[4] >= -84
 C7_trans_4_10: - 100 A_pick[4,10] - T_pick[4] + T_del[4] >= -89
 C7_trans_4_6: - 100 A_pick[4,6] - T_pick[4] + T_del[4] >= -89
 C7_trans_5_1: - 100 A_pick[5,1] - T_pick[5] + T_del[5] >= -86
 C7_trans_5_10: - 100 A_pick[5,10] - T_pick[5] + T_del[5] >= -87
 C7_trans_5_6: - 100 A_pick[5,6] - T_pick[5] + T_del[5] >= -91
 C8_between_2_1_1[1]: - 100 W_seq[2,1,1] - 100 A_pick[1,1] + T_pick[1]
   - T_end[2] >= -189
 C8_between_2_1_1[2]: - 100 W_seq[2,1,2] - 100 A_pick[1,1] + T_pick[1]
   - T_end[2] >= -189
 C8_between_2_1_1[3]: - 100 W_seq[2,1,3] - 100 A_pick[1,1] + T_pick[1]
   - T_end[2] >= -189
 C8_between_2_1_10[1]: - 100 W_seq[2,1,1] - 100 A_pick[1,10] + T_pick[1]
   - T_end[2] >= -184
 C8_between_2_1_10[2]: - 100 W_seq[2,1,2] - 100 A_pick[1,10] + T_pick[1]
   - T_end[2] >= -184
 C8_between_2_1_10[3]: - 100 W_seq[2,1,3] - 100 A_pick[1,10] + T_pick[1]
   - T_end[2] >= -184
 C8_between_2_1_6[1]: - 100 W_seq[2,1,1] - 100 A_pick[1,6] + T_pick[1]
   - T_end[2] >= -188
 C8_between_2_1_6[2]: - 100 W_seq[2,1,2] - 100 A_pick[1,6] + T_pick[1]
   - T_end[2] >= -188
 C8_between_2_1_6[3]: - 100 W_seq[2,1,3] - 100 A_pick[1,6] + T_pick[1]
   - T_end[2] >= -188
 C8_between_3_1_1[1]: - 100 W_seq[3,1,1] - 100 A_pick[1,1] + T_pick[1]
   - T_end[3] >= -189
 C8_between_3_1_1[2]: - 100 W_seq[3,1,2] - 100 A_pick[1,1] + T_pick[1]
   - T_end[3] >= -189
 C8_between_3_1_1[3]: - 100 W_seq[3,1,3] - 100 A_pick[1,1] + T_pick[1]
   - T_end[3] >= -189
 C8_between_3_1_10[1]: - 100 W_seq[3,1,1] - 100 A_pick[1,10] + T_pick[1]
   - T_end[3] >= -184
 C8_between_3_1_10[2]: - 100 W_seq[3,1,2] - 100 A_pick[1,10] + T_pick[1]
   - T_end[3] >= -184
 C8_between_3_1_10[3]: - 100 W_seq[3,1,3] - 100 A_pick[1,10] + T_pick[1]
   - T_end[3] >= -184
 C8_between_3_1_6[1]: - 100 W_seq[3,1,1] - 100 A_pick[1,6] + T_pick[1]
   - T_end[3] >= -188
 C8_between_3_1_6[2]: - 100 W_seq[3,1,2] - 100 A_pick[1,6] + T_pick[1]
   - T_end[3] >= -188
 C8_between_3_1_6[3]: - 100 W_seq[3,1,3] - 100 A_pick[1,6] + T_pick[1]
   - T_end[3] >= -188
 C8_between_4_1_1[1]: - 100 W_seq[4,1,1] - 100 A_pick[1,1] + T_pick[1]
   - T_end[4] >= -184
 C8_between_4_1_1[2]: - 100 W_seq[4,1,2] - 100 A_pick[1,1] + T_pick[1]
   - T_end[4] >= -184
 C8_between_4_1_1[3]: - 100 W_seq[4,1,3] - 100 A_pick[1,1] + T_pick[1]
   - T_end[4] >= -184
 C8_between_4_1_10[1]: - 100 W_seq[4,1,1] - 100 A_pick[1,10] + T_pick[1]
   - T_end[4] >= -189
 C8_between_4_1_10[2]: - 100 W_seq[4,1,2] - 100 A_pick[1,10] + T_pick[1]
   - T_end[4] >= -189
 C8_between_4_1_10[3]: - 100 W_seq[4,1,3] - 100 A_pick[1,10] + T_pick[1]
   - T_end[4] >= -189
 C8_between_4_1_6[1]: - 100 W_seq[4,1,1] - 100 A_pick[1,6] + T_pick[1]
   - T_end[4] >= -189
 C8_between_4_1_6[2]: - 100 W_seq[4,1,2] - 100 A_pick[1,6] + T_pick[1]
   - T_end[4] >= -189
 C8_between_4_1_6[3]: - 100 W_seq[4,1,3] - 100 A_pick[1,6] + T_pick[1]
   - T_end[4] >= -189
 C8_between_5_1_1[1]: - 100 W_seq[5,1,1] - 100 A_pick[1,1] + T_pick[1]
   - T_end[5] >= -186
 C8_between_5_1_1[2]: - 100 W_seq[5,1,2] - 100 A_pick[1,1] + T_pick[1]
   - T_end[5] >= -186
 C8_between_5_1_1[3]: - 100 W_seq[5,1,3] - 100 A_pick[1,1] + T_pick[1]
   - T_end[5] >= -186
 C8_between_5_1_10[1]: - 100 W_seq[5,1,1] - 100 A_pick[1,10] + T_pick[1]
   - T_end[5] >= -187
 C8_between_5_1_10[2]: - 100 W_seq[5,1,2] - 100 A_pick[1,10] + T_pick[1]
   - T_end[5] >= -187
 C8_between_5_1_10[3]: - 100 W_seq[5,1,3] - 100 A_pick[1,10] + T_pick[1]
   - T_end[5] >= -187
 C8_between_5_1_6[1]: - 100 W_seq[5,1,1] - 100 A_pick[1,6] + T_pick[1]
   - T_end[5] >= -191
 C8_between_5_1_6[2]: - 100 W_seq[5,1,2] - 100 A_pick[1,6] + T_pick[1]
   - T_end[5] >= -191
 C8_between_5_1_6[3]: - 100 W_seq[5,1,3] - 100 A_pick[1,6] + T_pick[1]
   - T_end[5] >= -191
 C8_start_1_1_1: - 100 W_seq[0,1,1] - 100 A_pick[1,1] + T_pick[1] >= -200
 C8_start_1_1_10: - 100 W_seq[0,1,1] - 100 A_pick[1,10] + T_pick[1]
   >= -191
 C8_start_1_1_6: - 100 W_seq[0,1,1] - 100 A_pick[1,6] + T_pick[1] >= -195
 C8_start_1_2_1: - 100 W_seq[0,1,2] - 100 A_pick[1,1] + T_pick[1] >= -195
 C8_start_1_2_10: - 100 W_seq[0,1,2] - 100 A_pick[1,10] + T_pick[1]
   >= -196
 C8_start_1_2_6: - 100 W_seq[0,1,2] - 100 A_pick[1,6] + T_pick[1] >= -200
 C8_start_1_3_1: - 100 W_seq[0,1,3] - 100 A_pick[1,1] + T_pick[1] >= -191
 C8_start_1_3_10: - 100 W_seq[0,1,3] - 100 A_pick[1,10] + T_pick[1]
   >= -200
 C8_start_1_3_6: - 100 W_seq[0,1,3] - 100 A_pick[1,6] + T_pick[1] >= -196
 C8_between_1_2_1[1]: - 100 W_seq[1,2,1] - 100 A_pick[2,1] + T_pick[2]
   - T_end[1] >= -184
 C8_between_1_2_1[2]: - 100 W_seq[1,2,2] - 100 A_pick[2,1] + T_pick[2]
   - T_end[1] >= -184
 C8_between_1_2_1[3]: - 100 W_seq[1,2,3] - 100 A_pick[2,1] + T_pick[2]
   - T_end[1] >= -184
 C8_between_1_2_10[1]: - 100 W_seq[1,2,1] - 100 A_pick[2,10] + T_pick[2]
   - T_end[1] >= -189
 C8_between_1_2_10[2]: - 100 W_seq[1,2,2] - 100 A_pick[2,10] + T_pick[2]
   - T_end[1] >= -189
 C8_between_1_2_10[3]: - 100 W_seq[1,2,3] - 100 A_pick[2,10] + T_pick[2]
   - T_end[1] >= -189
 C8_between_1_2_6[1]: - 100 W_seq[1,2,1] - 100 A_pick[2,6] + T_pick[2]
   - T_end[1] >= -189
 C8_between_1_2_6[2]: - 100 W_seq[1,2,2] - 100 A_pick[2,6] + T_pick[2]
   - T_end[1] >= -189
 C8_between_1_2_6[3]: - 100 W_seq[1,2,3] - 100 A_pick[2,6] + T_pick[2]
   - T_end[1] >= -189
 C8_between_3_2_1[1]: - 100 W_seq[3,2,1] - 100 A_pick[2,1] + T_pick[2]
   - T_end[3] >= -189
 C8_between_3_2_1[2]: - 100 W_seq[3,2,2] - 100 A_pick[2,1] + T_pick[2]
   - T_end[3] >= -189
 C8_between_3_2_1[3]: - 100 W_seq[3,2,3] - 100 A_pick[2,1] + T_pick[2]
   - T_end[3] >= -189
 C8_between_3_2_10[1]: - 100 W_seq[3,2,1] - 100 A_pick[2,10] + T_pick[2]
   - T_end[3] >= -184
 C8_between_3_2_10[2]: - 100 W_seq[3,2,2] - 100 A_pick[2,10] + T_pick[2]
   - T_end[3] >= -184
 C8_between_3_2_10[3]: - 100 W_seq[3,2,3] - 100 A_pick[2,10] + T_pick[2]
   - T_end[3] >= -184
 C8_between_3_2_6[1]: - 100 W_seq[3,2,1] - 100 A_pick[2,6] + T_pick[2]
   - T_end[3] >= -188
 C8_between_3_2_6[2]: - 100 W_seq[3,2,2] - 100 A_pick[2,6] + T_pick[2]
   - T_end[3] >= -188
 C8_between_3_2_6[3]: - 100 W_seq[3,2,3] - 100 A_pick[2,6] + T_pick[2]
   - T_end[3] >= -188
 C8_between_4_2_1[1]: - 100 W_seq[4,2,1] - 100 A_pick[2,1] + T_pick[2]
   - T_end[4] >= -184
 C8_between_4_2_1[2]: - 100 W_seq[4,2,2] - 100 A_pick[2,1] + T_pick[2]
   - T_end[4] >= -184
 C8_between_4_2_1[3]: - 100 W_seq[4,2,3] - 100 A_pick[2,1] + T_pick[2]
   - T_end[4] >= -184
 C8_between_4_2_10[1]: - 100 W_seq[4,2,1] - 100 A_pick[2,10] + T_pick[2]
   - T_end[4] >= -189
 C8_between_4_2_10[2]: - 100 W_seq[4,2,2] - 100 A_pick[2,10] + T_pick[2]
   - T_end[4] >= -189
 C8_between_4_2_10[3]: - 100 W_seq[4,2,3] - 100 A_pick[2,10] + T_pick[2]
   - T_end[4] >= -189
 C8_between_4_2_6[1]: - 100 W_seq[4,2,1] - 100 A_pick[2,6] + T_pick[2]
   - T_end[4] >= -189
 C8_between_4_2_6[2]: - 100 W_seq[4,2,2] - 100 A_pick[2,6] + T_pick[2]
   - T_end[4] >= -189
 C8_between_4_2_6[3]: - 100 W_seq[4,2,3] - 100 A_pick[2,6] + T_pick[2]
   - T_end[4] >= -189
 C8_between_5_2_1[1]: - 100 W_seq[5,2,1] - 100 A_pick[2,1] + T_pick[2]
   - T_end[5] >= -186
 C8_between_5_2_1[2]: - 100 W_seq[5,2,2] - 100 A_pick[2,1] + T_pick[2]
   - T_end[5] >= -186
 C8_between_5_2_1[3]: - 100 W_seq[5,2,3] - 100 A_pick[2,1] + T_pick[2]
   - T_end[5] >= -186
 C8_between_5_2_10[1]: - 100 W_seq[5,2,1] - 100 A_pick[2,10] + T_pick[2]
   - T_end[5] >= -187
 C8_between_5_2_10[2]: - 100 W_seq[5,2,2] - 100 A_pick[2,10] + T_pick[2]
   - T_end[5] >= -187
 C8_between_5_2_10[3]: - 100 W_seq[5,2,3] - 100 A_pick[2,10] + T_pick[2]
   - T_end[5] >= -187
 C8_between_5_2_6[1]: - 100 W_seq[5,2,1] - 100 A_pick[2,6] + T_pick[2]
   - T_end[5] >= -191
 C8_between_5_2_6[2]: - 100 W_seq[5,2,2] - 100 A_pick[2,6] + T_pick[2]
   - T_end[5] >= -191
 C8_between_5_2_6[3]: - 100 W_seq[5,2,3] - 100 A_pick[2,6] + T_pick[2]
   - T_end[5] >= -191
 C8_start_2_1_1: - 100 W_seq[0,2,1] - 100 A_pick[2,1] + T_pick[2] >= -200
 C8_start_2_1_10: - 100 W_seq[0,2,1] - 100 A_pick[2,10] + T_pick[2]
   >= -191
 C8_start_2_1_6: - 100 W_seq[0,2,1] - 100 A_pick[2,6] + T_pick[2] >= -195
 C8_start_2_2_1: - 100 W_seq[0,2,2] - 100 A_pick[2,1] + T_pick[2] >= -195
 C8_start_2_2_10: - 100 W_seq[0,2,2] - 100 A_pick[2,10] + T_pick[2]
   >= -196
 C8_start_2_2_6: - 100 W_seq[0,2,2] - 100 A_pick[2,6] + T_pick[2] >= -200
 C8_start_2_3_1: - 100 W_seq[0,2,3] - 100 A_pick[2,1] + T_pick[2] >= -191
 C8_start_2_3_10: - 100 W_seq[0,2,3] - 100 A_pick[2,10] + T_pick[2]
   >= -200
 C8_start_2_3_6: - 100 W_seq[0,2,3] - 100 A_pick[2,6] + T_pick[2] >= -196
 C8_between_1_3_1[1]: - 100 W_seq[1,3,1] - 100 A_pick[3,1] + T_pick[3]
   - T_end[1] >= -184
 C8_between_1_3_1[2]: - 100 W_seq[1,3,2] - 100 A_pick[3,1] + T_pick[3]
   - T_end[1] >= -184
 C8_between_1_3_1[3]: - 100 W_seq[1,3,3] - 100 A_pick[3,1] + T_pick[3]
   - T_end[1] >= -184
 C8_between_1_3_10[1]: - 100 W_seq[1,3,1] - 100 A_pick[3,10] + T_pick[3]
   - T_end[1] >= -189
 C8_between_1_3_10[2]: - 100 W_seq[1,3,2] - 100 A_pick[3,10] + T_pick[3]
   - T_end[1] >= -189
 C8_between_1_3_10[3]: - 100 W_seq[1,3,3] - 100 A_pick[3,10] + T_pick[3]
   - T_end[1] >= -189
 C8_between_1_3_6[1]: - 100 W_seq[1,3,1] - 100 A_pick[3,6] + T_pick[3]
   - T_end[1] >= -189
 C8_between_1_3_6[2]: - 100 W_seq[1,3,2] - 100 A_pick[3,6] + T_pick[3]
   - T_end[1] >= -189
 C8_between_1_3_6[3]: - 100 W_seq[1,3,3] - 100 A_pick[3,6] + T_pick[3]
   - T_end[1] >= -189
 C8_between_2_3_1[1]: - 100 W_seq[2,3,1] - 100 A_pick[3,1] + T_pick[3]
   - T_end[2] >= -189
 C8_between_2_3_1[2]: - 100 W_seq[2,3,2] - 100 A_pick[3,1] + T_pick[3]
   - T_end[2] >= -189
 C8_between_2_3_1[3]: - 100 W_seq[2,3,3] - 100 A_pick[3,1] + T_pick[3]
   - T_end[2] >= -189
 C8_between_2_3_10[1]: - 100 W_seq[2,3,1] - 100 A_pick[3,10] + T_pick[3]
   - T_end[2] >= -184
 C8_between_2_3_10[2]: - 100 W_seq[2,3,2] - 100 A_pick[3,10] + T_pick[3]
   - T_end[2] >= -184
 C8_between_2_3_10[3]: - 100 W_seq[2,3,3] - 100 A_pick[3,10] + T_pick[3]
   - T_end[2] >= -184
 C8_between_2_3_6[1]: - 100 W_seq[2,3,1] - 100 A_pick[3,6] + T_pick[3]
   - T_end[2] >= -188
 C8_between_2_3_6[2]: - 100 W_seq[2,3,2] - 100 A_pick[3,6] + T_pick[3]
   - T_end[2] >= -188
 C8_between_2_3_6[3]: - 100 W_seq[2,3,3] - 100 A_pick[3,6] + T_pick[3]
   - T_end[2] >= -188
 C8_between_4_3_1[1]: - 100 W_seq[4,3,1] - 100 A_pick[3,1] + T_pick[3]
   - T_end[4] >= -184
 C8_between_4_3_1[2]: - 100 W_seq[4,3,2] - 100 A_pick[3,1] + T_pick[3]
   - T_end[4] >= -184
 C8_between_4_3_1[3]: - 100 W_seq[4,3,3] - 100 A_pick[3,1] + T_pick[3]
   - T_end[4] >= -184
 C8_between_4_3_10[1]: - 100 W_seq[4,3,1] - 100 A_pick[3,10] + T_pick[3]
   - T_end[4] >= -189
 C8_between_4_3_10[2]: - 100 W_seq[4,3,2] - 100 A_pick[3,10] + T_pick[3]
   - T_end[4] >= -189
 C8_between_4_3_10[3]: - 100 W_seq[4,3,3] - 100 A_pick[3,10] + T_pick[3]
   - T_end[4] >= -189
 C8_between_4_3_6[1]: - 100 W_seq[4,3,1] - 100 A_pick[3,6] + T_pick[3]
   - T_end[4] >= -189
 C8_between_4_3_6[2]: - 100 W_seq[4,3,2] - 100 A_pick[3,6] + T_pick[3]
   - T_end[4] >= -189
 C8_between_4_3_6[3]: - 100 W_seq[4,3,3] - 100 A_pick[3,6] + T_pick[3]
   - T_end[4] >= -189
 C8_between_5_3_1[1]: - 100 W_seq[5,3,1] - 100 A_pick[3,1] + T_pick[3]
   - T_end[5] >= -186
 C8_between_5_3_1[2]: - 100 W_seq[5,3,2] - 100 A_pick[3,1] + T_pick[3]
   - T_end[5] >= -186
 C8_between_5_3_1[3]: - 100 W_seq[5,3,3] - 100 A_pick[3,1] + T_pick[3]
   - T_end[5] >= -186
 C8_between_5_3_10[1]: - 100 W_seq[5,3,1] - 100 A_pick[3,10] + T_pick[3]
   - T_end[5] >= -187
 C8_between_5_3_10[2]: - 100 W_seq[5,3,2] - 100 A_pick[3,10] + T_pick[3]
   - T_end[5] >= -187
 C8_between_5_3_10[3]: - 100 W_seq[5,3,3] - 100 A_pick[3,10] + T_pick[3]
   - T_end[5] >= -187
 C8_between_5_3_6[1]: - 100 W_seq[5,3,1] - 100 A_pick[3,6] + T_pick[3]
   - T_end[5] >= -191
 C8_between_5_3_6[2]: - 100 W_seq[5,3,2] - 100 A_pick[3,6] + T_pick[3]
   - T_end[5] >= -191
 C8_between_5_3_6[3]: - 100 W_seq[5,3,3] - 100 A_pick[3,6] + T_pick[3]
   - T_end[5] >= -191
 C8_start_3_1_1: - 100 W_seq[0,3,1] - 100 A_pick[3,1] + T_pick[3] >= -200
 C8_start_3_1_10: - 100 W_seq[0,3,1] - 100 A_pick[3,10] + T_pick[3]
   >= -191
 C8_start_3_1_6: - 100 W_seq[0,3,1] - 100 A_pick[3,6] + T_pick[3] >= -195
 C8_start_3_2_1: - 100 W_seq[0,3,2] - 100 A_pick[3,1] + T_pick[3] >= -195
 C8_start_3_2_10: - 100 W_seq[0,3,2] - 100 A_pick[3,10] + T_pick[3]
   >= -196
 C8_start_3_2_6: - 100 W_seq[0,3,2] - 100 A_pick[3,6] + T_pick[3] >= -200
 C8_start_3_3_1: - 100 W_seq[0,3,3] - 100 A_pick[3,1] + T_pick[3] >= -191
 C8_start_3_3_10: - 100 W_seq[0,3,3] - 100 A_pick[3,10] + T_pick[3]
   >= -200
 C8_start_3_3_6: - 100 W_seq[0,3,3] - 100 A_pick[3,6] + T_pick[3] >= -196
 C8_between_1_4_1[1]: - 100 W_seq[1,4,1] - 100 A_pick[4,1] + T_pick[4]
   - T_end[1] >= -184
 C8_between_1_4_1[2]: - 100 W_seq[1,4,2] - 100 A_pick[4,1] + T_pick[4]
   - T_end[1] >= -184
 C8_between_1_4_1[3]: - 100 W_seq[1,4,3] - 100 A_pick[4,1] + T_pick[4]
   - T_end[1] >= -184
 C8_between_1_4_10[1]: - 100 W_seq[1,4,1] - 100 A_pick[4,10] + T_pick[4]
   - T_end[1] >= -189
 C8_between_1_4_10[2]: - 100 W_seq[1,4,2] - 100 A_pick[4,10] + T_pick[4]
   - T_end[1] >= -189
 C8_between_1_4_10[3]: - 100 W_seq[1,4,3] - 100 A_pick[4,10] + T_pick[4]
   - T_end[1] >= -189
 C8_between_1_4_6[1]: - 100 W_seq[1,4,1] - 100 A_pick[4,6] + T_pick[4]
   - T_end[1] >= -189
 C8_between_1_4_6[2]: - 100 W_seq[1,4,2] - 100 A_pick[4,6] + T_pick[4]
   - T_end[1] >= -189
 C8_between_1_4_6[3]: - 100 W_seq[1,4,3] - 100 A_pick[4,6] + T_pick[4]
   - T_end[1] >= -189
 C8_between_2_4_1[1]: - 100 W_seq[2,4,1] - 100 A_pick[4,1] + T_pick[4]
   - T_end[2] >= -189
 C8_between_2_4_1[2]: - 100 W_seq[2,4,2] - 100 A_pick[4,1] + T_pick[4]
   - T_end[2] >= -189
 C8_between_2_4_1[3]: - 100 W_seq[2,4,3] - 100 A_pick[4,1] + T_pick[4]
   - T_end[2] >= -189
 C8_between_2_4_10[1]: - 100 W_seq[2,4,1] - 100 A_pick[4,10] + T_pick[4]
   - T_end[2] >= -184
 C8_between_2_4_10[2]: - 100 W_seq[2,4,2] - 100 A_pick[4,10] + T_pick[4]
   - T_end[2] >= -184
 C8_between_2_4_10[3]: - 100 W_seq[2,4,3] - 100 A_pick[4,10] + T_pick[4]
   - T_end[2] >= -184
 C8_between_2_4_6[1]: - 100 W_seq[2,4,1] - 100 A_pick[4,6] + T_pick[4]
   - T_end[2] >= -188
 C8_between_2_4_6[2]: - 100 W_seq[2,4,2] - 100 A_pick[4,6] + T_pick[4]
   - T_end[2] >= -188
 C8_between_2_4_6[3]: - 100 W_seq[2,4,3] - 100 A_pick[4,6] + T_pick[4]
   - T_end[2] >= -188
 C8_between_3_4_1[1]: - 100 W_seq[3,4,1] - 100 A_pick[4,1] + T_pick[4]
   - T_end[3] >= -189
 C8_between_3_4_1[2]: - 100 W_seq[3,4,2] - 100 A_pick[4,1] + T_pick[4]
   - T_end[3] >= -189
 C8_between_3_4_1[3]: - 100 W_seq[3,4,3] - 100 A_pick[4,1] + T_pick[4]
   - T_end[3] >= -189
 C8_between_3_4_10[1]: - 100 W_seq[3,4,1] - 100 A_pick[4,10] + T_pick[4]
   - T_end[3] >= -184
 C8_between_3_4_10[2]: - 100 W_seq[3,4,2] - 100 A_pick[4,10] + T_pick[4]
   - T_end[3] >= -184
 C8_between_3_4_10[3]: - 100 W_seq[3,4,3] - 100 A_pick[4,10] + T_pick[4]
   - T_end[3] >= -184
 C8_between_3_4_6[1]: - 100 W_seq[3,4,1] - 100 A_pick[4,6] + T_pick[4]
   - T_end[3] >= -188
 C8_between_3_4_6[2]: - 100 W_seq[3,4,2] - 100 A_pick[4,6] + T_pick[4]
   - T_end[3] >= -188
 C8_between_3_4_6[3]: - 100 W_seq[3,4,3] - 100 A_pick[4,6] + T_pick[4]
   - T_end[3] >= -188
 C8_between_5_4_1[1]: - 100 W_seq[5,4,1] - 100 A_pick[4,1] + T_pick[4]
   - T_end[5] >= -186
 C8_between_5_4_1[2]: - 100 W_seq[5,4,2] - 100 A_pick[4,1] + T_pick[4]
   - T_end[5] >= -186
 C8_between_5_4_1[3]: - 100 W_seq[5,4,3] - 100 A_pick[4,1] + T_pick[4]
   - T_end[5] >= -186
 C8_between_5_4_10[1]: - 100 W_seq[5,4,1] - 100 A_pick[4,10] + T_pick[4]
   - T_end[5] >= -187
 C8_between_5_4_10[2]: - 100 W_seq[5,4,2] - 100 A_pick[4,10] + T_pick[4]
   - T_end[5] >= -187
 C8_between_5_4_10[3]: - 100 W_seq[5,4,3] - 100 A_pick[4,10] + T_pick[4]
   - T_end[5] >= -187
 C8_between_5_4_6[1]: - 100 W_seq[5,4,1] - 100 A_pick[4,6] + T_pick[4]
   - T_end[5] >= -191
 C8_between_5_4_6[2]: - 100 W_seq[5,4,2] - 100 A_pick[4,6] + T_pick[4]
   - T_end[5] >= -191
 C8_between_5_4_6[3]: - 100 W_seq[5,4,3] - 100 A_pick[4,6] + T_pick[4]
   - T_end[5] >= -191
 C8_start_4_1_1: - 100 W_seq[0,4,1] - 100 A_pick[4,1] + T_pick[4] >= -200
 C8_start_4_1_10: - 100 W_seq[0,4,1] - 100 A_pick[4,10] + T_pick[4]
   >= -191
 C8_start_4_1_6: - 100 W_seq[0,4,1] - 100 A_pick[4,6] + T_pick[4] >= -195
 C8_start_4_2_1: - 100 W_seq[0,4,2] - 100 A_pick[4,1] + T_pick[4] >= -195
 C8_start_4_2_10: - 100 W_seq[0,4,2] - 100 A_pick[4,10] + T_pick[4]
   >= -196
 C8_start_4_2_6: - 100 W_seq[0,4,2] - 100 A_pick[4,6] + T_pick[4] >= -200
 C8_start_4_3_1: - 100 W_seq[0,4,3] - 100 A_pick[4,1] + T_pick[4] >= -191
 C8_start_4_3_10: - 100 W_seq[0,4,3] - 100 A_pick[4,10] + T_pick[4]
   >= -200
 C8_start_4_3_6: - 100 W_seq[0,4,3] - 100 A_pick[4,6] + T_pick[4] >= -196
 C8_between_1_5_1[1]: - 100 W_seq[1,5,1] - 100 A_pick[5,1] + T_pick[5]
   - T_end[1] >= -184
 C8_between_1_5_1[2]: - 100 W_seq[1,5,2] - 100 A_pick[5,1] + T_pick[5]
   - T_end[1] >= -184
 C8_between_1_5_1[3]: - 100 W_seq[1,5,3] - 100 A_pick[5,1] + T_pick[5]
   - T_end[1] >= -184
 C8_between_1_5_10[1]: - 100 W_seq[1,5,1] - 100 A_pick[5,10] + T_pick[5]
   - T_end[1] >= -189
 C8_between_1_5_10[2]: - 100 W_seq[1,5,2] - 100 A_pick[5,10] + T_pick[5]
   - T_end[1] >= -189
 C8_between_1_5_10[3]: - 100 W_seq[1,5,3] - 100 A_pick[5,10] + T_pick[5]
   - T_end[1] >= -189
 C8_between_1_5_6[1]: - 100 W_seq[1,5,1] - 100 A_pick[5,6] + T_pick[5]
   - T_end[1] >= -189
 C8_between_1_5_6[2]: - 100 W_seq[1,5,2] - 100 A_pick[5,6] + T_pick[5]
   - T_end[1] >= -189
 C8_between_1_5_6[3]: - 100 W_seq[1,5,3] - 100 A_pick[5,6] + T_pick[5]
   - T_end[1] >= -189
 C8_between_2_5_1[1]: - 100 W_seq[2,5,1] - 100 A_pick[5,1] + T_pick[5]
   - T_end[2] >= -189
 C8_between_2_5_1[2]: - 100 W_seq[2,5,2] - 100 A_pick[5,1] + T_pick[5]
   - T_end[2] >= -189
 C8_between_2_5_1[3]: - 100 W_seq[2,5,3] - 100 A_pick[5,1] + T_pick[5]
   - T_end[2] >= -189
 C8_between_2_5_10[1]: - 100 W_seq[2,5,1] - 100 A_pick[5,10] + T_pick[5]
   - T_end[2] >= -184
 C8_between_2_5_10[2]: - 100 W_seq[2,5,2] - 100 A_pick[5,10] + T_pick[5]
   - T_end[2] >= -184
 C8_between_2_5_10[3]: - 100 W_seq[2,5,3] - 100 A_pick[5,10] + T_pick[5]
   - T_end[2] >= -184
 C8_between_2_5_6[1]: - 100 W_seq[2,5,1] - 100 A_pick[5,6] + T_pick[5]
   - T_end[2] >= -188
 C8_between_2_5_6[2]: - 100 W_seq[2,5,2] - 100 A_pick[5,6] + T_pick[5]
   - T_end[2] >= -188
 C8_between_2_5_6[3]: - 100 W_seq[2,5,3] - 100 A_pick[5,6] + T_pick[5]
   - T_end[2] >= -188
 C8_between_3_5_1[1]: - 100 W_seq[3,5,1] - 100 A_pick[5,1] + T_pick[5]
   - T_end[3] >= -189
 C8_between_3_5_1[2]: - 100 W_seq[3,5,2] - 100 A_pick[5,1] + T_pick[5]
   - T_end[3] >= -189
 C8_between_3_5_1[3]: - 100 W_seq[3,5,3] - 100 A_pick[5,1] + T_pick[5]
   - T_end[3] >= -189
 C8_between_3_5_10[1]: - 100 W_seq[3,5,1] - 100 A_pick[5,10] + T_pick[5]
   - T_end[3] >= -184
 C8_between_3_5_10[2]: - 100 W_seq[3,5,2] - 100 A_pick[5,10] + T_pick[5]
   - T_end[3] >= -184
 C8_between_3_5_10[3]: - 100 W_seq[3,5,3] - 100 A_pick[5,10] + T_pick[5]
   - T_end[3] >= -184
 C8_between_3_5_6[1]: - 100 W_seq[3,5,1] - 100 A_pick[5,6] + T_pick[5]
   - T_end[3] >= -188
 C8_between_3_5_6[2]: - 100 W_seq[3,5,2] - 100 A_pick[5,6] + T_pick[5]
   - T_end[3] >= -188
 C8_between_3_5_6[3]: - 100 W_seq[3,5,3] - 100 A_pick[5,6] + T_pick[5]
   - T_end[3] >= -188
 C8_between_4_5_1[1]: - 100 W_seq[4,5,1] - 100 A_pick[5,1] + T_pick[5]
   - T_end[4] >= -184
 C8_between_4_5_1[2]: - 100 W_seq[4,5,2] - 100 A_pick[5,1] + T_pick[5]
   - T_end[4] >= -184
 C8_between_4_5_1[3]: - 100 W_seq[4,5,3] - 100 A_pick[5,1] + T_pick[5]
   - T_end[4] >= -184
 C8_between_4_5_10[1]: - 100 W_seq[4,5,1] - 100 A_pick[5,10] + T_pick[5]
   - T_end[4] >= -189
 C8_between_4_5_10[2]: - 100 W_seq[4,5,2] - 100 A_pick[5,10] + T_pick[5]
   - T_end[4] >= -189
 C8_between_4_5_10[3]: - 100 W_seq[4,5,3] - 100 A_pick[5,10] + T_pick[5]
   - T_end[4] >= -189
 C8_between_4_5_6[1]: - 100 W_seq[4,5,1] - 100 A_pick[5,6] + T_pick[5]
   - T_end[4] >= -189
 C8_between_4_5_6[2]: - 100 W_seq[4,5,2] - 100 A_pick[5,6] + T_pick[5]
   - T_end[4] >= -189
 C8_between_4_5_6[3]: - 100 W_seq[4,5,3] - 100 A_pick[5,6] + T_pick[5]
   - T_end[4] >= -189
 C8_start_5_1_1: - 100 W_seq[0,5,1] - 100 A_pick[5,1] + T_pick[5] >= -200
 C8_start_5_1_10: - 100 W_seq[0,5,1] - 100 A_pick[5,10] + T_pick[5]
   >= -191
 C8_start_5_1_6: - 100 W_seq[0,5,1] - 100 A_pick[5,6] + T_pick[5] >= -195
 C8_start_5_2_1: - 100 W_seq[0,5,2] - 100 A_pick[5,1] + T_pick[5] >= -195
 C8_start_5_2_10: - 100 W_seq[0,5,2] - 100 A_pick[5,10] + T_pick[5]
   >= -196
 C8_start_5_2_6: - 100 W_seq[0,5,2] - 100 A_pick[5,6] + T_pick[5] >= -200
 C8_start_5_3_1: - 100 W_seq[0,5,3] - 100 A_pick[5,1] + T_pick[5] >= -191
 C8_start_5_3_10: - 100 W_seq[0,5,3] - 100 A_pick[5,10] + T_pick[5]
   >= -200
 C8_start_5_3_6: - 100 W_seq[0,5,3] - 100 A_pick[5,6] + T_pick[5] >= -196
Bounds
Binaries
 Y_assign[1,1] Y_assign[1,2] Y_assign[1,3] Y_assign[2,1] Y_assign[2,2]
 Y_assign[2,3] Y_assign[3,1] Y_assign[3,2] Y_assign[3,3] Y_assign[4,1]
 Y_assign[4,2] Y_assign[4,3] Y_assign[5,1] Y_assign[5,2] Y_assign[5,3]
 W_seq[0,0,1] W_seq[0,0,2] W_seq[0,0,3] W_seq[0,1,1] W_seq[0,1,2]
 W_seq[0,1,3] W_seq[0,2,1] W_seq[0,2,2] W_seq[0,2,3] W_seq[0,3,1]
 W_seq[0,3,2] W_seq[0,3,3] W_seq[0,4,1] W_seq[0,4,2] W_seq[0,4,3]
 W_seq[0,5,1] W_seq[0,5,2] W_seq[0,5,3] W_seq[0,6,1] W_seq[0,6,2]
 W_seq[0,6,3] W_seq[1,0,1] W_seq[1,0,2] W_seq[1,0,3] W_seq[1,1,1]
 W_seq[1,1,2] W_seq[1,1,3] W_seq[1,2,1] W_seq[1,2,2] W_seq[1,2,3]
 W_seq[1,3,1] W_seq[1,3,2] W_seq[1,3,3] W_seq[1,4,1] W_seq[1,4,2]
 W_seq[1,4,3] W_seq[1,5,1] W_seq[1,5,2] W_seq[1,5,3] W_seq[1,6,1]
 W_seq[1,6,2] W_seq[1,6,3] W_seq[2,0,1] W_seq[2,0,2] W_seq[2,0,3]
 W_seq[2,1,1] W_seq[2,1,2] W_seq[2,1,3] W_seq[2,2,1] W_seq[2,2,2]
 W_seq[2,2,3] W_seq[2,3,1] W_seq[2,3,2] W_seq[2,3,3] W_seq[2,4,1]
 W_seq[2,4,2] W_seq[2,4,3] W_seq[2,5,1] W_seq[2,5,2] W_seq[2,5,3]
 W_seq[2,6,1] W_seq[2,6,2] W_seq[2,6,3] W_seq[3,0,1] W_seq[3,0,2]
 W_seq[3,0,3] W_seq[3,1,1] W_seq[3,1,2] W_seq[3,1,3] W_seq[3,2,1]
 W_seq[3,2,2] W_seq[3,2,3] W_seq[3,3,1] W_seq[3,3,2] W_seq[3,3,3]
 W_seq[3,4,1] W_seq[3,4,2] W_seq[3,4,3] W_seq[3,5,1] W_seq[3,5,2]
 W_seq[3,5,3] W_seq[3,6,1] W_seq[3,6,2] W_seq[3,6,3] W_seq[4,0,1]
 W_seq[4,0,2] W_seq[4,0,3] W_seq[4,1,1] W_seq[4,1,2] W_seq[4,1,3]
 W_seq[4,2,1] W_seq[4,2,2] W_seq[4,2,3] W_seq[4,3,1] W_seq[4,3,2]
 W_seq[4,3,3] W_seq[4,4,1] W_seq[4,4,2] W_seq[4,4,3] W_seq[4,5,1]
 W_seq[4,5,2] W_seq[4,5,3] W_seq[4,6,1] W_seq[4,6,2] W_seq[4,6,3]
 W_seq[5,0,1] W_seq[5,0,2] W_seq[5,0,3] W_seq[5,1,1] W_seq[5,1,2]
 W_seq[5,1,3] W_seq[5,2,1] W_seq[5,2,2] W_seq[5,2,3] W_seq[5,3,1]
 W_seq[5,3,2] W_seq[5,3,3] W_seq[5,4,1] W_seq[5,4,2] W_seq[5,4,3]
 W_seq[5,5,1] W_seq[5,5,2] W_seq[5,5,3] W_seq[5,6,1] W_seq[5,6,2]
 W_seq[5,6,3] W_seq[6,0,1] W_seq[6,0,2] W_seq[6,0,3] W_seq[6,1,1]
 W_seq[6,1,2] W_seq[6,1,3] W_seq[6,2,1] W_seq[6,2,2] W_seq[6,2,3]
 W_seq[6,3,1] W_seq[6,3,2] W_seq[6,3,3] W_seq[6,4,1] W_seq[6,4,2]
 W_seq[6,4,3] W_seq[6,5,1] W_seq[6,5,2] W_seq[6,5,3] W_seq[6,6,1]
 W_seq[6,6,2] W_seq[6,6,3] A_pick[1,1] A_pick[1,10] A_pick[1,6] A_pick[2,1]
 A_pick[2,10] A_pick[2,6] A_pick[3,1] A_pick[3,10] A_pick[3,6] A_pick[4,1]
 A_pick[4,10] A_pick[4,6] A_pick[5,1] A_pick[5,10] A_pick[5,6]
End
