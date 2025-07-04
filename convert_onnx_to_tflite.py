import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import os

# --- 設定項目 ---
ONNX_MODEL_PATH = 'best.onnx'
TFLITE_MODEL_PATH = 'best.tflite'
# -----------------

print(f"ONNXモデル '{ONNX_MODEL_PATH}' を読み込んでいます...")
try:
    onnx_model = onnx.load(ONNX_MODEL_PATH)
    tf_rep = prepare(onnx_model)
    
    print("中間形式 (SavedModel) としてエクスポートします...")
    tf_rep.export_graph('saved_model')
    
    print("SavedModelをTFLite形式に変換します...")
    converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
    
    # ★★★★★【最終修正点】★★★★★
    # TFLiteの標準機能で扱えない処理（TF Select ops）も、TensorFlowの機能を使って変換できるようにする
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # TFLiteの標準命令セット
        tf.lite.OpsSet.SELECT_TF_OPS    # TensorFlowの命令セットも許可
    ]
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
        
    print(f"\n✅ 変換完了！ TFLiteモデルを '{TFLITE_MODEL_PATH}' に保存しました。")

except FileNotFoundError:
    print(f"エラー: ONNXモデル '{ONNX_MODEL_PATH}' が見つかりません。")
    print("先に `export_to_onnx.py` を実行して、ONNXファイルを生成してください。")
except Exception as e:
    print(f"変換中にエラーが発生しました: {e}")