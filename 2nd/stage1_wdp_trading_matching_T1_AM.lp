\ Model stage1_wdp_trading_matching_T1_AM
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xc737fb30f09337b4
Maximize
  1970 q[O1,S1,P1] + 1650 q[O1,S2,P1] + 2110 q[O1,S3,P1] + 1776 q[O1,S1,P2]
   + 1960 q[O1,S4,P2] + 1750 q[O2,S1,P1] + 1465 q[O2,S2,P1]
   + 1915 q[O2,S3,P1] + 1182.5 q[O2,S2,P3] + 1976 q[O5,S1,P2]
   + 2160 q[O5,S4,P2] + 2392 q[O5,S3,P4] + 2240 q[O5,S4,P4]
   - 3000 w_buyer_seller[B1,S1] - 5000 w_buyer_seller[B1,S2]
   - 9000 w_buyer_seller[B1,S3] - 5000 w_buyer_seller[B1,S4]
   - 5000 w_buyer_seller[B2,S1] - 3500 w_buyer_seller[B2,S2]
   - 8500 w_buyer_seller[B2,S3] - 3000 w_buyer_seller[B4,S1]
   - 9000 w_buyer_seller[B4,S3] - 5000 w_buyer_seller[B4,S4]
Subject To
 total_fulfilled_O1_P1: - q[O1,S1,P1] - q[O1,S2,P1] - q[O1,S3,P1]
   + Q[O1,P1] = 0
 total_fulfilled_O1_P2: - q[O1,S1,P2] - q[O1,S4,P2] + Q[O1,P2] = 0
 total_fulfilled_O2_P1: - q[O2,S1,P1] - q[O2,S2,P1] - q[O2,S3,P1]
   + Q[O2,P1] = 0
 total_fulfilled_O2_P3: - q[O2,S2,P3] + Q[O2,P3] = 0
 total_fulfilled_O5_P2: - q[O5,S1,P2] - q[O5,S4,P2] + Q[O5,P2] = 0
 total_fulfilled_O5_P4: - q[O5,S3,P4] - q[O5,S4,P4] + Q[O5,P4] = 0
 min_fill_O1_P1: - 40 x_order[O1] + Q[O1,P1] >= 0
 min_fill_O1_P2: - 24 x_order[O1] + Q[O1,P2] >= 0
 min_fill_O2_P1: - 32 x_order[O2] + Q[O2,P1] >= 0
 min_fill_O2_P3: - 48 x_order[O2] + Q[O2,P3] >= 0
 min_fill_O5_P2: - 28 x_order[O5] + Q[O5,P2] >= 0
 min_fill_O5_P4: - 20 x_order[O5] + Q[O5,P4] >= 0
 max_demand_O1_P1: - 50 x_order[O1] + Q[O1,P1] <= 0
 max_demand_O1_P2: - 30 x_order[O1] + Q[O1,P2] <= 0
 max_demand_O2_P1: - 40 x_order[O2] + Q[O2,P1] <= 0
 max_demand_O2_P3: - 60 x_order[O2] + Q[O2,P3] <= 0
 max_demand_O5_P2: - 35 x_order[O5] + Q[O5,P2] <= 0
 max_demand_O5_P4: - 25 x_order[O5] + Q[O5,P4] <= 0
 inventory_S1_P1: q[O1,S1,P1] + q[O2,S1,P1] <= 100
 inventory_S1_P2: q[O1,S1,P2] + q[O5,S1,P2] <= 50
 inventory_S2_P1: q[O1,S2,P1] + q[O2,S2,P1] <= 80
 inventory_S2_P3: q[O2,S2,P3] <= 120
 inventory_S3_P1: q[O1,S3,P1] + q[O2,S3,P1] <= 90
 inventory_S3_P4: q[O5,S3,P4] <= 70
 inventory_S4_P2: q[O1,S4,P2] + q[O5,S4,P2] <= 100
 inventory_S4_P4: q[O5,S4,P4] <= 60
 link_q_z_O1_S1_P1: q[O1,S1,P1] - 50 z[O1,S1,P1] <= 0
 link_q_z_O1_S2_P1: q[O1,S2,P1] - 50 z[O1,S2,P1] <= 0
 link_q_z_O1_S3_P1: q[O1,S3,P1] - 50 z[O1,S3,P1] <= 0
 link_q_z_O1_S1_P2: q[O1,S1,P2] - 30 z[O1,S1,P2] <= 0
 link_q_z_O1_S4_P2: q[O1,S4,P2] - 30 z[O1,S4,P2] <= 0
 link_q_z_O2_S1_P1: q[O2,S1,P1] - 40 z[O2,S1,P1] <= 0
 link_q_z_O2_S2_P1: q[O2,S2,P1] - 40 z[O2,S2,P1] <= 0
 link_q_z_O2_S3_P1: q[O2,S3,P1] - 40 z[O2,S3,P1] <= 0
 link_q_z_O2_S2_P3: q[O2,S2,P3] - 60 z[O2,S2,P3] <= 0
 link_q_z_O5_S1_P2: q[O5,S1,P2] - 35 z[O5,S1,P2] <= 0
 link_q_z_O5_S4_P2: q[O5,S4,P2] - 35 z[O5,S4,P2] <= 0
 link_q_z_O5_S3_P4: q[O5,S3,P4] - 25 z[O5,S3,P4] <= 0
 link_q_z_O5_S4_P4: q[O5,S4,P4] - 25 z[O5,S4,P4] <= 0
 link_z_x_O1_S1_P1: - x_order[O1] + z[O1,S1,P1] <= 0
 link_z_x_O1_S2_P1: - x_order[O1] + z[O1,S2,P1] <= 0
 link_z_x_O1_S3_P1: - x_order[O1] + z[O1,S3,P1] <= 0
 link_z_x_O1_S1_P2: - x_order[O1] + z[O1,S1,P2] <= 0
 link_z_x_O1_S4_P2: - x_order[O1] + z[O1,S4,P2] <= 0
 link_z_x_O2_S1_P1: - x_order[O2] + z[O2,S1,P1] <= 0
 link_z_x_O2_S2_P1: - x_order[O2] + z[O2,S2,P1] <= 0
 link_z_x_O2_S3_P1: - x_order[O2] + z[O2,S3,P1] <= 0
 link_z_x_O2_S2_P3: - x_order[O2] + z[O2,S2,P3] <= 0
 link_z_x_O5_S1_P2: - x_order[O5] + z[O5,S1,P2] <= 0
 link_z_x_O5_S4_P2: - x_order[O5] + z[O5,S4,P2] <= 0
 link_z_x_O5_S3_P4: - x_order[O5] + z[O5,S3,P4] <= 0
 link_z_x_O5_S4_P4: - x_order[O5] + z[O5,S4,P4] <= 0
 link_z_y_O1_S1_P1: z[O1,S1,P1] - y_seller_sku[S1,P1] <= 0
 link_z_y_O1_S2_P1: z[O1,S2,P1] - y_seller_sku[S2,P1] <= 0
 link_z_y_O1_S3_P1: z[O1,S3,P1] - y_seller_sku[S3,P1] <= 0
 link_z_y_O1_S1_P2: z[O1,S1,P2] - y_seller_sku[S1,P2] <= 0
 link_z_y_O1_S4_P2: z[O1,S4,P2] - y_seller_sku[S4,P2] <= 0
 link_z_y_O2_S1_P1: z[O2,S1,P1] - y_seller_sku[S1,P1] <= 0
 link_z_y_O2_S2_P1: z[O2,S2,P1] - y_seller_sku[S2,P1] <= 0
 link_z_y_O2_S3_P1: z[O2,S3,P1] - y_seller_sku[S3,P1] <= 0
 link_z_y_O2_S2_P3: z[O2,S2,P3] - y_seller_sku[S2,P3] <= 0
 link_z_y_O5_S1_P2: z[O5,S1,P2] - y_seller_sku[S1,P2] <= 0
 link_z_y_O5_S4_P2: z[O5,S4,P2] - y_seller_sku[S4,P2] <= 0
 link_z_y_O5_S3_P4: z[O5,S3,P4] - y_seller_sku[S3,P4] <= 0
 link_z_y_O5_S4_P4: z[O5,S4,P4] - y_seller_sku[S4,P4] <= 0
 min_lot_ratio_S1_P1: q[O1,S1,P1] + q[O2,S1,P1] - 20 y_seller_sku[S1,P1]
   >= 0
 min_lot_ratio_S1_P2: q[O1,S1,P2] + q[O5,S1,P2] - 10 y_seller_sku[S1,P2]
   >= 0
 min_lot_ratio_S2_P1: q[O1,S2,P1] + q[O2,S2,P1] - 16 y_seller_sku[S2,P1]
   >= 0
 min_lot_ratio_S2_P3: q[O2,S2,P3] - 24 y_seller_sku[S2,P3] >= 0
 min_lot_ratio_S3_P1: q[O1,S3,P1] + q[O2,S3,P1] - 18 y_seller_sku[S3,P1]
   >= 0
 min_lot_ratio_S3_P4: q[O5,S3,P4] - 14 y_seller_sku[S3,P4] >= 0
 min_lot_ratio_S4_P2: q[O1,S4,P2] + q[O5,S4,P2] - 20 y_seller_sku[S4,P2]
   >= 0
 min_lot_ratio_S4_P4: q[O5,S4,P4] - 12 y_seller_sku[S4,P4] >= 0
 lot_usage_upper_S1_P1: q[O1,S1,P1] + q[O2,S1,P1] - 100 y_seller_sku[S1,P1]
   <= 0
 lot_usage_upper_S1_P2: q[O1,S1,P2] + q[O5,S1,P2] - 50 y_seller_sku[S1,P2]
   <= 0
 lot_usage_upper_S2_P1: q[O1,S2,P1] + q[O2,S2,P1] - 80 y_seller_sku[S2,P1]
   <= 0
 lot_usage_upper_S2_P3: q[O2,S2,P3] - 120 y_seller_sku[S2,P3] <= 0
 lot_usage_upper_S3_P1: q[O1,S3,P1] + q[O2,S3,P1] - 90 y_seller_sku[S3,P1]
   <= 0
 lot_usage_upper_S3_P4: q[O5,S3,P4] - 70 y_seller_sku[S3,P4] <= 0
 lot_usage_upper_S4_P2: q[O1,S4,P2] + q[O5,S4,P2] - 100 y_seller_sku[S4,P2]
   <= 0
 lot_usage_upper_S4_P4: q[O5,S4,P4] - 60 y_seller_sku[S4,P4] <= 0
 budget_O1: - 500000 x_order[O1] + 6000 Q[O1,P1] + 7000 Q[O1,P2] <= 0
 budget_O2: - 400000 x_order[O2] + 5800 Q[O2,P1] + 4200 Q[O2,P3] <= 0
 budget_O5: - 450000 x_order[O5] + 7200 Q[O5,P2] + 8500 Q[O5,P4] <= 0
 link_z_w_O1_S1_P1: z[O1,S1,P1] - w_buyer_seller[B1,S1] <= 0
 link_z_w_O1_S2_P1: z[O1,S2,P1] - w_buyer_seller[B1,S2] <= 0
 link_z_w_O1_S3_P1: z[O1,S3,P1] - w_buyer_seller[B1,S3] <= 0
 link_z_w_O1_S1_P2: z[O1,S1,P2] - w_buyer_seller[B1,S1] <= 0
 link_z_w_O1_S4_P2: z[O1,S4,P2] - w_buyer_seller[B1,S4] <= 0
 link_z_w_O2_S1_P1: z[O2,S1,P1] - w_buyer_seller[B2,S1] <= 0
 link_z_w_O2_S2_P1: z[O2,S2,P1] - w_buyer_seller[B2,S2] <= 0
 link_z_w_O2_S3_P1: z[O2,S3,P1] - w_buyer_seller[B2,S3] <= 0
 link_z_w_O2_S2_P3: z[O2,S2,P3] - w_buyer_seller[B2,S2] <= 0
 link_z_w_O5_S1_P2: z[O5,S1,P2] - w_buyer_seller[B4,S1] <= 0
 link_z_w_O5_S4_P2: z[O5,S4,P2] - w_buyer_seller[B4,S4] <= 0
 link_z_w_O5_S3_P4: z[O5,S3,P4] - w_buyer_seller[B4,S3] <= 0
 link_z_w_O5_S4_P4: z[O5,S4,P4] - w_buyer_seller[B4,S4] <= 0
 link_w_z_B1_S1: - z[O1,S1,P1] - z[O1,S1,P2] + w_buyer_seller[B1,S1] <= 0
 link_w_z_B1_S2: - z[O1,S2,P1] + w_buyer_seller[B1,S2] <= 0
 link_w_z_B1_S3: - z[O1,S3,P1] + w_buyer_seller[B1,S3] <= 0
 link_w_z_B1_S4: - z[O1,S4,P2] + w_buyer_seller[B1,S4] <= 0
 link_w_z_B2_S1: - z[O2,S1,P1] + w_buyer_seller[B2,S1] <= 0
 link_w_z_B2_S2: - z[O2,S2,P1] - z[O2,S2,P3] + w_buyer_seller[B2,S2] <= 0
 link_w_z_B2_S3: - z[O2,S3,P1] + w_buyer_seller[B2,S3] <= 0
 link_w_z_B4_S1: - z[O5,S1,P2] + w_buyer_seller[B4,S1] <= 0
 link_w_z_B4_S3: - z[O5,S3,P4] + w_buyer_seller[B4,S3] <= 0
 link_w_z_B4_S4: - z[O5,S4,P2] - z[O5,S4,P4] + w_buyer_seller[B4,S4] <= 0
 non_deficit_estimated_delivery_cost: 1970 q[O1,S1,P1] + 1650 q[O1,S2,P1]
   + 2110 q[O1,S3,P1] + 1776 q[O1,S1,P2] + 1960 q[O1,S4,P2]
   + 1750 q[O2,S1,P1] + 1465 q[O2,S2,P1] + 1915 q[O2,S3,P1]
   + 1182.5 q[O2,S2,P3] + 1976 q[O5,S1,P2] + 2160 q[O5,S4,P2]
   + 2392 q[O5,S3,P4] + 2240 q[O5,S4,P4] - 3000 w_buyer_seller[B1,S1]
   - 5000 w_buyer_seller[B1,S2] - 9000 w_buyer_seller[B1,S3]
   - 5000 w_buyer_seller[B1,S4] - 5000 w_buyer_seller[B2,S1]
   - 3500 w_buyer_seller[B2,S2] - 8500 w_buyer_seller[B2,S3]
   - 3000 w_buyer_seller[B4,S1] - 9000 w_buyer_seller[B4,S3]
   - 5000 w_buyer_seller[B4,S4] >= 0
Bounds
Binaries
 x_order[O1] x_order[O2] x_order[O5] z[O1,S1,P1] z[O1,S2,P1] z[O1,S3,P1]
 z[O1,S1,P2] z[O1,S4,P2] z[O2,S1,P1] z[O2,S2,P1] z[O2,S3,P1] z[O2,S2,P3]
 z[O5,S1,P2] z[O5,S4,P2] z[O5,S3,P4] z[O5,S4,P4] y_seller_sku[S1,P1]
 y_seller_sku[S1,P2] y_seller_sku[S2,P1] y_seller_sku[S2,P3]
 y_seller_sku[S3,P1] y_seller_sku[S3,P4] y_seller_sku[S4,P2]
 y_seller_sku[S4,P4] w_buyer_seller[B1,S1] w_buyer_seller[B1,S2]
 w_buyer_seller[B1,S3] w_buyer_seller[B1,S4] w_buyer_seller[B2,S1]
 w_buyer_seller[B2,S2] w_buyer_seller[B2,S3] w_buyer_seller[B4,S1]
 w_buyer_seller[B4,S3] w_buyer_seller[B4,S4]
End
