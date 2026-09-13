# Results

Thư mục này chứa kết quả thực nghiệm của các mô hình baseline và mô hình đề xuất.

## Files

| File | Mô tả |
|------|--------|
| `baseline_results.csv` | Bảng tổng hợp MAE/RMSE/MAPE của 5 baseline |
| `agcrn_pems04.json` | Kết quả chi tiết AGCRN trên PeMS04 |
| `agcrn_pems08.json` | Kết quả chi tiết AGCRN trên PeMS08 |
| `dcrnn_pems04.json` | Kết quả chi tiết DCRNN trên PeMS04 |
| `dcrnn_pems08.json` | Kết quả chi tiết DCRNN trên PeMS08 |
| `stfgnn_pems04.json` | Kết quả chi tiết STFGNN trên PeMS04 |
| `stfgnn_pems08.json` | Kết quả chi tiết STFGNN trên PeMS08 |
| `pdformer_pems04.json` | Kết quả chi tiết PDFormer trên PeMS04 |
| `pdformer_pems08.json` | Kết quả chi tiết PDFormer trên PeMS08 |
| `stdmae_pems04.json` | Kết quả chi tiết STD-MAE trên PeMS04 |
| `stdmae_pems08.json` | Kết quả chi tiết STD-MAE trên PeMS08 |

## Metrics

- **MAE**: Mean Absolute Error (thấp hơn = tốt hơn)  
- **RMSE**: Root Mean Squared Error (thấp hơn = tốt hơn)
- **MAPE**: Mean Absolute Percentage Error % (thấp hơn = tốt hơn)

Kết quả tại 3 mốc dự báo: **15 phút** (horizon 3), **30 phút** (horizon 6), **60 phút** (horizon 12).

## Lệnh chạy lại

```bash
# Chạy tất cả baseline (50 epoch)
python scripts/run_all_baselines.py --epochs 50

# Chỉ thu thập kết quả (không train lại)
python scripts/run_all_baselines.py --collect-only

# Smoke test pipeline
python scripts/run_all_baselines.py --smoke-test
```
