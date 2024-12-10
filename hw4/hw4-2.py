# 安裝與導入所需函式庫
import pandas as pd
from pycaret.classification import *
import optuna
import xgboost
import catboost

# 步驟 1：數據加載與特徵工程
train_data = pd.read_csv(r'C:\Users\v52no\OneDrive\桌面\hw4\train.csv')
test_data = pd.read_csv(r'C:\Users\v52no\OneDrive\桌面\hw4\test.csv')

# 處理缺失值
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# 新增特徵工程
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch']
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

# 編碼類別型特徵
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# 刪除無用特徵
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_ids = test_data['PassengerId']
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# 步驟 2：初始化 PyCaret 環境
exp = setup(data=train_data, target='Survived', session_id=123, use_gpu=False)

# 步驟 3：模型比較與選擇
best_model = compare_models()

# 步驟 4：模型集成
ensemble_model = blend_models(estimator_list=[best_model], fold=5)

# 步驟 5：超參數優化
tuned_model = tune_model(ensemble_model)

# 步驟 6：對測試集進行預測
final_model = finalize_model(tuned_model)
predictions = predict_model(final_model, data=test_data)

# 檢查返回的 DataFrame 結構
print(predictions.head())  # 查看欄位名稱

# 提取預測結果
if 'Label' in predictions.columns:
    survived_predictions = predictions['Label']
elif 'prediction_label' in predictions.columns:
    survived_predictions = predictions['prediction_label']
else:
    raise KeyError("預測結果中找不到 'Label' 或 'prediction_label' 欄位，請檢查輸出結構。")

# 步驟 7：生成提交檔案
submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': survived_predictions})
submission.to_csv(r'C:\Users\v52no\OneDrive\桌面\hw4\submission_hw4_2.csv', index=False)

print("提交文件已生成，保存於桌面 hw4 資料夾中：submission_hw4_2.csv")
