from ultralytics import YOLO

# 訓練済みモデルのパス
model_path = 'best.pt'

print(f"モデル '{model_path}' を読み込んでいます...")
model = YOLO(model_path)

print("モデルをONNX (.onnx) 形式に変換します...")
# ★★★フォーマットを 'onnx' に指定★★★
model.export(format='onnx')

print("\n✅ 変換完了！")
print("同じ階層に 'best.onnx' というファイルが作成されました。")