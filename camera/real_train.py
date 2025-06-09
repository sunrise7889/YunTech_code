#讀取CSV做SVM訓練後輸出模型
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. 讀取資料
data = pd.read_csv("camera\svm_dataset.csv", header=None)
data.columns = ["hand_x", "hand_y", "prev_hand_x", "prev_hand_y", "ball_x", "ball_y", "label"]

X = data[["hand_x", "hand_y", "prev_hand_x", "prev_hand_y", "ball_x", "ball_y"]]
y = data["label"]

# 2. 切分訓練 / 測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 建立 SVM 模型（不使用標準化）
model = SVC(kernel='rbf', C=1.0, gamma='scale')

# 4. 訓練模型
model.fit(X_train, y_train)

# 5. 預測 + 評估
y_pred = model.predict(X_test)
print("準確率:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 6. 儲存模型
joblib.dump(model, "svm_model_no_scaler.pkl")
