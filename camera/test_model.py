import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === 設定檔案路徑（你可以換成其他CSV）===
csv_path = "camera/test.csv"
model_path = "camera/svm_model.pkl"
output_csv_path = "camera/svm_prediction_result.csv"

# === 讀取測試資料 ===
df = pd.read_csv(csv_path, header=None)
df.columns = ["hand_x", "hand_y", "prev_hand_x", "prev_hand_y", "ball_x", "ball_y", "label"]

X = df[["hand_x", "hand_y", "prev_hand_x", "prev_hand_y", "ball_x", "ball_y"]]
y_true = df["label"]

# === 載入模型（整合標準化器與 SVM）===
model = joblib.load(model_path)
y_pred = model.predict(X)

# === 評估結果 ===
print("==== 模型準確率 ====")
print(f"{accuracy_score(y_true, y_pred):.4f}\\n")

print("==== 分類報告 ====")
print(classification_report(y_true, y_pred))

# === 混淆矩陣圖 ===
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Predict 0", "Predict 1"],
            yticklabels=["Real 0", "Real 1"])
plt.xlabel("Predict")
plt.ylabel("Real")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# === 散佈圖 ===
correct = y_pred == y_true
incorrect = ~correct

plt.figure(figsize=(10, 6))
plt.scatter(df["ball_x"][correct], df["ball_y"][correct], c='green', label='Success', alpha=0.7)
plt.scatter(df["ball_x"][incorrect], df["ball_y"][incorrect], c='red', label='Fail', alpha=0.7)
plt.xlabel("Ball X")
plt.ylabel("Ball Y")
plt.title("Predicted Distribution")
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# === 將預測結果寫入 CSV ===
df["predicted"] = y_pred
df["result"] = ["Success" if t == p else "Fail" for t, p in zip(y_true, y_pred)]
df.to_csv(output_csv_path, index=False)

print(f"\n✅ 驗證結果已儲存到：{output_csv_path}")
