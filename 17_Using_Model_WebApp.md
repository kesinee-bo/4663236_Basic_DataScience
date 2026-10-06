# ตัวอย่างการประยุกต์ใช้โมเดล Classification กับ Web Application

Oct 6, 2026 · @Lost Star

## ภาพรวม

Train โมเดล **KNN** จำแนกผู้ป่วยเบาหวานจากไฟล์ `diabetes_clean.csv` บน Colab วัดผลด้วย 5-fold cross-validation (Accuracy, Precision, Recall, F1-Score โดยดู F1-Score เป็นหลัก) แล้วนำไปให้หน้าเว็บ JavaScript ธรรมดาเรียกใช้ 2 แบบ: **แบบ A** ผ่าน FastAPI และ **แบบ B** รันในเบราว์เซอร์ด้วย onnxruntime-web ทั้งสองแบบให้ผลทำนายเท่ากัน และทดสอบรันครบทุกขั้นแล้ว

![สถาปัตยกรรม · train 1 ครั้ง ใช้งาน 2 แบบ](architecture.png)

แบบ A มีเซิร์ฟเวอร์ Python คั่นกลาง ส่วนแบบ B ส่งไฟล์โมเดลไปให้เบราว์เซอร์รันเอง หน้าเว็บสองแบบใช้ HTML/CSS ชุดเดียวกัน ต่างกันแค่ `app.js`

| ไฟล์ใน zip | ใช้ทำอะไร |
| --- | --- |
| `diabetes_clean.csv` | dataset ที่ clean แล้ว 768 แถว สำหรับอัปโหลดเข้า Colab |
| `06_diabetes.ipynb, 06_diabetes_backend_api.ipynb` | notebook 1: อ่านไฟล์, train KNN, วัดผล, export · notebook 2: รัน API บน Colab + ngrok |
| `backend/main.py`, `requirements.txt` | API แบบ A สำหรับ**รันบนเครื่องตัวเอง**เท่านั้น ถ้ารัน API บน Colab (notebook 2) ไม่ต้องใช้ |
| `web-api/` | หน้าเว็บแบบ A: index.html (แบบเต็ม มีช่อง API URL) และ predict.html (ตัวอย่างสั้นจาก notebook 2 ข้อ 3.6) |
| `web-onnx/` | หน้าเว็บแบบ B + ไฟล์ `.onnx` ตัวอย่าง |
| `README.md` | คำสั่งรันทั้งหมดแบบย่อ |

## Dataset: Pima Indians Diabetes

ใช้ไฟล์ `diabetes_clean.csv` (แนบมาใน zip) ซึ่งเป็นข้อมูลชุด [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) จาก Kaggle: 768 แถว, 8 feature, เป้าหมาย `Outcome` (1 = เป็นเบาหวาน 268 คน หรือ 34.9%) เป็นงาน binary classification ขนาดเล็กที่ train บน Colab ได้ในไม่กี่วินาที

| คอลัมน์ | ความหมาย | หน่วย |
| --- | --- | --- |
| Pregnancies | จำนวนครั้งที่ตั้งครรภ์ | ครั้ง |
| Glucose | น้ำตาลในเลือดหลังดื่มน้ำตาล 2 ชม. (OGTT) | mg/dL |
| BloodPressure | ความดันตัวล่าง (diastolic) | mmHg |
| SkinThickness | ความหนาผิวหนังต้นแขน (triceps) | mm |
| Insulin | อินซูลินในเลือด 2 ชม. | μU/mL |
| BMI | ดัชนีมวลกาย | kg/m² |
| DiabetesPedigreeFunction | ค่าความเสี่ยงจากประวัติครอบครัว | – |
| Age | อายุ | ปี |

**นำไฟล์เข้า Colab** ดูโค้ดในหัวข้อถัดไป ขั้นที่ 2

**เตรียมข้อมูลไว้แล้ว** ในข้อมูลต้นฉบับ 5 คอลัมน์ Glucose, BloodPressure, SkinThickness, Insulin และ BMI มีค่า 0 ที่เป็นไปไม่ได้ทางการแพทย์ (เช่น Insulin = 0 มี 374 แถว) ไฟล์ `diabetes_clean.csv` ที่แนบมาแทนค่า 0 เหล่านี้ด้วยค่ามัธยฐานของคอลัมน์แล้ว ไม่มีค่าว่าง จึงนำไป train ได้ทันที

| คอลัมน์ | ค่า 0 ในต้นฉบับ | แทนด้วยค่ามัธยฐาน |
| --- | --- | --- |
| Glucose | 5 แถว | 117 |
| BloodPressure | 35 แถว | 72 |
| SkinThickness | 227 แถว | 29 |
| Insulin | 374 แถว | 125 |
| BMI | 11 แถว | 32.3 |

## Notebook 1 (06\_diabetes.ipynb): Train และ export โมเดล

งานบน Colab แบ่งเป็น 2 notebook: **notebook 1** train และ export โมเดล ส่วน **notebook 2** รัน API ให้หน้าเว็บเรียก เปิด Colab สร้าง notebook ใหม่ (File → New notebook) แล้วสร้าง cell ทีละ cell ตามลำดับด้านล่าง วางโค้ดแต่ละกล่องลงใน cell ของตัวเอง แล้วกด Shift + Enter ต้องรันเรียงลำดับเพราะ cell หลังใช้ตัวแปรจาก cell ก่อนหน้า ใต้โค้ดบางกล่องมีผลลัพธ์ที่ควรเห็นไว้เทียบ (หรืออัปโหลด `06_diabetes.ipynb` และ `06_diabetes_backend_api.ipynb` จาก zip ที่มีทุก cell ให้แล้ว)

### ขั้นที่ 1: ตรวจเวอร์ชันไลบรารี

Colab ติดตั้ง scikit-learn, pandas, numpy และ joblib มาให้แล้ว cell นี้ import และพิมพ์เวอร์ชันออกมา ให้จดเลขเวอร์ชัน scikit-learn ไว้ใส่ใน `backend/requirements.txt` เพราะเวอร์ชันตอน train กับตอนโหลดบนเซิร์ฟเวอร์ต้องตรงกัน

```python
import sklearn, pandas as pd, numpy as np, joblib
print("scikit-learn :", sklearn.__version__)
print("pandas       :", pd.__version__)
print("joblib       :", joblib.__version__)
```

### ขั้นที่ 2: อัปโหลดและโหลดข้อมูล

รัน cell แล้วกดปุ่ม Choose Files เลือกไฟล์ `diabetes_clean.csv` จากเครื่อง ถ้าลากไฟล์ไปวางในแถบ Files ด้านซ้ายไว้ก่อนแล้ว cell จะไม่ถามซ้ำ (ไฟล์ที่อัปโหลดจะหายเมื่อเปิด session ใหม่)

ไฟล์นี้ clean มาแล้ว จึงแยก `X` (8 feature) กับ `y` (Outcome) ได้ทันที ลำดับคอลัมน์ใน `FEATURES` สำคัญ เพราะหน้าเว็บต้องส่งข้อมูลเรียงตามนี้

```python
import os

csv_path = "diabetes_clean.csv"
if not os.path.exists(csv_path):
    from google.colab import files
    uploaded = files.upload()                 # เลือกไฟล์ diabetes_clean.csv
    csv_path = next(iter(uploaded))           # ใช้ชื่อไฟล์ที่อัปโหลดจริง

df = pd.read_csv(csv_path)

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
TARGET = "Outcome"

X = df[FEATURES]   # 8 feature
y = df[TARGET]     # 0 = ไม่เป็นเบาหวาน, 1 = เป็นเบาหวาน

print("X:", X.shape, " y:", y.shape)
print(y.value_counts())
df.head()
```

ควรได้ `X: (768, 8)  y: (768,)` และคลาส 0 = 500 คน, คลาส 1 = 268 คน

### ขั้นที่ 3: วัดผล KNN ด้วย 5-fold Cross-Validation

ใช้โมเดล **KNN (K-Nearest Neighbors)** ตัวเดียว KNN ทายผลจาก 15 คนในข้อมูลที่ค่าใกล้เคียงที่สุด (เพื่อนบ้าน) ถ้าในนั้นเป็นเบาหวาน 11 คน ความน่าจะเป็นคือ 11/15 = 73.3% ก่อน train ต้อง **scale** ข้อมูลด้วย `StandardScaler` ให้ทุกคอลัมน์อยู่ในสเกลเดียวกัน เพราะ KNN วัดระยะทาง ถ้าไม่ scale คอลัมน์ที่ค่าใหญ่อย่าง Insulin จะมีผลเกินจริง โค้ดเขียนทุกขั้นตอนออกมาตรงๆ ไม่ใช้ `Pipeline` เพื่อให้เห็นว่าเกิดอะไรขึ้นบ้าง

**5-fold cross-validation** แบ่งข้อมูล 768 แถวเป็น 5 ส่วน (`StratifiedKFold` รักษาสัดส่วนคลาส) แต่ละรอบทำ 3 ขั้น แล้วเฉลี่ยผล 5 รอบ (คลาสบวก = เป็นเบาหวาน)

1. **scale:** เรียนค่าเฉลี่ยและส่วนเบี่ยงเบนจากชุด train (4 ส่วน) เท่านั้น แล้วใช้แปลงทั้งชุด train และ test ถ้าเรียนจากข้อมูลทั้งหมดจะเท่ากับแอบเห็นชุด test ก่อน
2. **train KNN** ด้วยชุด train ที่ scale แล้ว
3. **ทำนายชุด test** (ส่วนที่เหลือ) แล้ววัดผล 4 ตัวชี้วัด

| ตัวชี้วัด | ความหมาย |
| --- | --- |
| Accuracy | ทายถูกกี่ % จากทั้งหมด |
| Precision | ที่ทายว่า "เป็นเบาหวาน" ถูกจริงกี่ % |
| Recall | คนที่เป็นเบาหวานจริง ทายเจอกี่ % |
| **F1-Score (ตัวหลัก)** | ค่าเฉลี่ยฮาร์มอนิกของ Precision กับ Recall |

ดู **F1-Score เป็นหลัก** เพราะคลาสไม่สมดุล (เป็นเบาหวานแค่ 35%) Accuracy อาจดูสูงได้แม้ทายเจอผู้ป่วยน้อย ส่วน F1-Score จะสูงได้ก็ต่อเมื่อทั้ง Precision และ Recall สูงพร้อมกัน

```python
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODEL_NAME = "KNN"
K = 15                                             # จำนวนเพื่อนบ้านที่ใช้ทำนาย

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
y_cv_pred = np.zeros(len(y), dtype=int)            # เก็บผลทำนายทุกแถว ไว้ทำ confusion matrix ในขั้นที่ 4

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 1) scale: เรียนจากชุด train เท่านั้น แล้วใช้แปลงทั้ง train และ test
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 2) train KNN
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train_s, y_train)

    # 3) ทำนายชุด test แล้ววัดผล
    pred = knn.predict(X_test_s)
    y_cv_pred[test_idx] = pred

    rows.append({
        "accuracy":  accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall":    recall_score(y_test, pred),
        "f1":        f1_score(y_test, pred),
    })

# ผลแต่ละรอบ + ค่าเฉลี่ย 5 รอบ
folds_df = pd.DataFrame(rows, index=[f"รอบที่ {i}" for i in range(1, 6)])
cv_mean = folds_df.mean().to_dict()
folds_df.loc["ค่าเฉลี่ย"] = folds_df.mean()
folds_df.round(3)
```

ผลที่ควรเห็น (scikit-learn 1.9.1 อาจต่างเล็กน้อยถ้าเวอร์ชันใน Colab ต่างไป)

| รอบ | Accuracy | Precision | Recall | **F1-Score** |
| --- | --- | --- | --- | --- |
| รอบที่ 1 | 0.753 | 0.682 | 0.556 | **0.612** |
| รอบที่ 2 | 0.779 | 0.717 | 0.611 | **0.660** |
| รอบที่ 3 | 0.727 | 0.630 | 0.537 | **0.580** |
| รอบที่ 4 | 0.765 | 0.707 | 0.547 | **0.617** |
| รอบที่ 5 | 0.732 | 0.620 | 0.585 | **0.602** |
| **ค่าเฉลี่ย** | **0.751** | **0.671** | **0.567** | **0.614** |

แต่ละรอบได้ผลต่างกันเล็กน้อย เพราะทดสอบกับข้อมูลคนละกลุ่ม (เช่น F1-Score 0.580 ถึง 0.660) ค่าเฉลี่ย 5 รอบจึงน่าเชื่อถือกว่าการวัดครั้งเดียว และค่าเฉลี่ยนี้คือ `cv_mean` ที่บันทึกลง `model_meta.json` ในขั้นที่ 5 F1-Score เฉลี่ย 0.614 ถูกดึงลงมาโดย Recall (ทายเจอผู้ป่วยจริงราว 57%) ถ้าอยากให้ F1 สูงขึ้น ลองเปลี่ยนค่า `K` แล้วรันใหม่เทียบกันได้

### ขั้นที่ 4: Confusion Matrix + Train ด้วยข้อมูลทั้งหมด

`y_cv_pred` จากขั้นที่ 3 เก็บผลทำนายของทุกแถวในรอบที่แถวนั้นเป็นข้อมูลทดสอบ จึงสร้าง confusion matrix จากทั้ง 768 แถวได้ จากนั้น scale และ train KNN ใหม่ด้วยข้อมูลทั้งหมด เพราะ CV มีไว้วัดผล ส่วนโมเดลที่ export ควรได้เรียนจากข้อมูลครบทุกแถว

```python
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_predictions(y, y_cv_pred, display_labels=["Negative", "Positive"])
plt.title(f"Confusion Matrix (5-fold CV) - {MODEL_NAME}")
plt.show()

# train ด้วยข้อมูลทั้งหมด เพื่อ export
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)                 # 1) scale

knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_scaled, y)                               # 2) train KNN
print("train เสร็จ:", knn)
```

ผลของ KNN ที่ควรเห็น

|  | ทายว่าไม่เป็น | ทายว่าเป็น |
| --- | --- | --- |
| **ไม่เป็นจริง (500)** | 425 | 75 |
| **เป็นจริง (268)** | 116 | 152 |

KNN ทายเจอผู้ป่วยจริง 152 จาก 268 คน (Recall 0.57) และพลาดไป 116 คน

### ขั้นที่ 5: (แบบ A) Export โมเดล (.joblib) + model\_meta.json

บันทึก 2 ไฟล์ลงโฟลเดอร์ทำงานของ Colab (เห็นในแถบ Files ด้านซ้าย)

- `diabetes_model.joblib` เก็บ `scaler` และ `knn` คู่กันในไฟล์เดียว เพราะตอนทำนายต้องใช้ทั้งสองตัว (scale ก่อน แล้วค่อยทำนาย) ใช้กับ FastAPI ในแบบ A
- `model_meta.json` เก็บชื่อ feature ตามลำดับ, ชื่อคลาส, ผลวัด 4 ตัว, ค่า mean/scale ของ scaler (หน้าเว็บแบบ B ใช้ scale ข้อมูลเอง) และเวอร์ชัน scikit-learn

```python
import json, datetime

joblib.dump({"scaler": scaler, "model": knn}, "diabetes_model.joblib")

meta = {
    "model_name": MODEL_NAME,
    "k": K,
    "features": FEATURES,
    "classes": {"0": "ไม่เป็นเบาหวาน", "1": "เป็นเบาหวาน"},
    "evaluation": "5-fold stratified cross-validation (ค่าเฉลี่ย 5 รอบ)",
    "metrics": {k: round(float(v), 4) for k, v in cv_mean.items()},
    # ค่าของ StandardScaler: x_scaled = (x - mean) / scale  (เรียงตาม FEATURES)
    "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
    "sklearn_version": sklearn.__version__,
    "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
}
with open("model_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(json.dumps(meta["metrics"], indent=2))
```

ควรได้

```json
{
  "accuracy": 0.7513,
  "precision": 0.6714,
  "recall": 0.5672,
  "f1": 0.6142
}
```

### ขั้นที่ 6: (แบบ B) Export เป็น ONNX (.onnx)

ไฟล์ `.joblib` ใช้ได้เฉพาะใน Python ถ้าจะให้ JavaScript ทำนายเองในเบราว์เซอร์ ต้องแปลง **KNN** เป็น ONNX ด้วย `skl2onnx` ซึ่ง Colab ไม่ได้ติดตั้งมาให้ บรรทัดแรกจึงติดตั้งก่อน ไฟล์ `.onnx` มีแค่ KNN ส่วนการ scale หน้าเว็บทำเองด้วยค่า mean/scale ใน `model_meta.json`

- `initial_types` กำหนด input ชื่อ `input` เป็น float32 ขนาด `[จำนวนแถว, 8]` ข้อมูลต้อง scale แล้ว ฝั่ง JavaScript ต้องส่งข้อมูลชื่อนี้และเรียงคอลัมน์ตาม `FEATURES`
- `zipmap: False` ทำให้ output `probabilities` เป็น array ธรรมดา อ่านใน JavaScript ได้ง่าย
- ท้าย cell รันทั้ง 768 แถวผ่าน onnxruntime เทียบกับ scikit-learn เพื่อยืนยันว่าแปลงถูกต้อง

```python
!pip -q install skl2onnx onnxruntime

from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

onnx_model = to_onnx(
    knn,
    initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
    options={id(knn): {"zipmap": False}},          # probabilities เป็น array
    target_opset=17,
)
with open("diabetes_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# ตรวจว่า ONNX ให้ผลตรงกับ scikit-learn (ส่งข้อมูลที่ scale แล้ว)
import onnxruntime as ort
sess = ort.InferenceSession("diabetes_model.onnx")
onnx_label, onnx_prob = sess.run(None, {"input": X_scaled.astype(np.float32)})
print("label ตรงกัน:", (onnx_label == knn.predict(X_scaled)).mean() * 100, "%")
```

ควรได้ `label ตรงกัน: 100.0 %`

### ขั้นที่ 7: ดาวน์โหลดไฟล์ไปใช้

cell นี้ดาวน์โหลดทั้ง 3 ไฟล์ลงเครื่อง (เบราว์เซอร์อาจถามอนุญาตให้ดาวน์โหลดหลายไฟล์ ให้กดอนุญาต) หรือคลิกขวาที่ไฟล์ในแถบ Files แล้วเลือก Download ก็ได้

```python
from google.colab import files
files.download("diabetes_model.joblib")
files.download("diabetes_model.onnx")
files.download("model_meta.json")
```

| ไฟล์ | ใช้ที่ |
| --- | --- |
| `diabetes_model.joblib` | อัปโหลดเข้า notebook 2 (แบบ A บน Colab) หรือวางใน `backend/` (แบบ A บนเครื่อง) |
| `model_meta.json` | อัปโหลดเข้า notebook 2 และวางใน `web-onnx/` (หรือ `backend/`) |
| `diabetes_model.onnx` | `web-onnx/` (แบบ B) |

## Notebook 2 (06\_diabetes\_backend\_api.ipynb): รัน API บน Colab

notebook นี้แยกจาก notebook 1 ใช้แค่ 2 ไฟล์ที่ดาวน์โหลดมาจากขั้นที่ 7 คือ `diabetes_model.joblib` และ `model_meta.json` เปิดเป็น notebook ใหม่ใน Colab แล้วรันตามลำดับ cell ที่ 1 ตรวจเวอร์ชันไลบรารีแบบเดียวกับขั้นที่ 1 ของ notebook 1

### ขั้นที่ 1: อัปโหลดไฟล์โมเดล

notebook 2 เป็นคนละ session กับ notebook 1 ไฟล์ที่สร้างไว้ใน notebook 1 จึงไม่มีอยู่ที่นี่ รัน cell แล้วเลือกทั้ง 2 ไฟล์พร้อมกัน (หรือลากไฟล์ไปวางในแถบ Files ก่อน cell จะไม่ถามซ้ำ) ถ้าไฟล์ยังไม่ครบ cell จะหยุดพร้อมบอกว่าขาดไฟล์ไหน

```python
import os

NEEDED = ["diabetes_model.joblib", "model_meta.json"]
missing = [f for f in NEEDED if not os.path.exists(f)]
if missing:
    from google.colab import files
    print("เลือกไฟล์:", ", ".join(missing))
    files.upload()

missing = [f for f in NEEDED if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"ยังไม่มีไฟล์ {missing} (ชื่อไฟล์ต้องตรงทุกตัวอักษร ถ้า Colab ตั้งชื่อเป็น '... (1)' ให้เปลี่ยนชื่อก่อน)")
print("พร้อมใช้งาน:", NEEDED)
```

ควรได้ `พร้อมใช้งาน: ['diabetes_model.joblib', 'model_meta.json']`

### ขั้นที่ 2: โหลดโมเดลกลับมาทดลองทำนาย

cell นี้จำลองสิ่งที่ backend ทำทุกครั้งที่มีคำขอ: โหลด `scaler` และ `knn` จากไฟล์ `.joblib` และ `model_meta.json` → รับข้อมูลผู้ป่วย 1 คนเป็น dict → แปลงเป็น DataFrame เรียงคอลัมน์ตาม `FEATURES` → **scale** ด้วยค่าเดียวกับตอน train → ทำนาย และเช็กว่าเวอร์ชัน scikit-learn ตรงกับตอน train ใน notebook 1 ถ้าไม่ตรงจะขึ้นคำเตือน ตัวแปร `scaler`, `knn`, `meta`, `FEATURES` และ `patient` จาก cell นี้ใช้ต่อในขั้นที่ 3

```python
import json
import joblib
import pandas as pd

# โหลดโมเดล: ไฟล์เก็บ scaler และ knn คู่กัน
bundle = joblib.load("diabetes_model.joblib")
scaler = bundle["scaler"]
knn = bundle["model"]

# โหลด metadata (ชื่อคลาส และ FEATURES สำหรับเรียงลำดับคอลัมน์)
with open("model_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
FEATURES = meta["features"]

# เวอร์ชัน scikit-learn ตอน train (notebook 1) กับตอนนี้ควรตรงกัน
if meta["sklearn_version"] != sklearn.__version__:
    print(f"คำเตือน: โมเดล train ด้วย scikit-learn {meta['sklearn_version']} "
          f"แต่ตอนนี้ใช้ {sklearn.__version__} ผลอาจเพี้ยนหรือโหลดไม่ได้")

patient = {"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35,
           "Insulin": 125, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}

X_new = pd.DataFrame([patient], columns=FEATURES)   # ลำดับคอลัมน์ตรงกับตอน train
X_new_s = scaler.transform(X_new)                   # 1) scale ด้วยค่าเดียวกับตอน train
pred = int(knn.predict(X_new_s)[0])                 # 2) ทำนาย
prob = float(knn.predict_proba(X_new_s)[0, 1])
print(f"ผลทำนาย: {pred} ({meta['classes'][str(pred)]})  ความน่าจะเป็นเป็นเบาหวาน = {prob:.1%}")
```

ควรได้ `ผลทำนาย: 1 (เป็นเบาหวาน)  ความน่าจะเป็นเป็นเบาหวาน = 73.3%`

### ขั้นที่ 3: รัน API บน Colab แล้วเปิดให้เว็บเรียกผ่าน ngrok

รัน FastAPI ค้างไว้ใน Colab แทนการรันบนเครื่องตัวเอง ข้อดีคือไม่ต้องติดตั้ง Python บนเครื่อง และใช้ scikit-learn เวอร์ชันเดียวกับตอน train แน่นอน แต่เซิร์ฟเวอร์ใน Colab เข้าจากภายนอกไม่ได้ **ngrok** จึงสร้าง URL สาธารณะ (`https://xxxx.ngrok-free.app`) ที่ส่งต่อคำขอมายังพอร์ต 8000 ใน Colab

**เตรียมก่อน (ทำครั้งเดียว)**

1. สมัครฟรีที่ [ngrok](https://dashboard.ngrok.com/signup) แล้วคัดลอก Authtoken จากเมนู Your Authtoken
2. ใน Colab คลิกไอคอนกุญแจ (Secrets) แถบซ้าย → Add new secret ตั้งชื่อ `NGROK_AUTHTOKEN` วาง token แล้วเปิดสวิตช์ Notebook access (ถ้าไม่ตั้ง cell จะถามให้วาง token แทน)

**3.1 ติดตั้งไลบรารี**

```python
!pip -q install fastapi uvicorn pyngrok
```

**3.2 สร้าง API ใน cell นี้เลย** เขียนโค้ด FastAPI ลงใน cell ของ Colab ตรงๆ ใช้ `scaler`, `knn`, `meta` และ `FEATURES` ที่โหลดไว้ในขั้นที่ 2 endpoint `/predict` ทำขั้นตอนเดียวกับขั้นที่ 2: เรียงคอลัมน์ → scale → ทำนาย

- `PatientInput` บอกว่า API รับข้อมูลอะไร Pydantic ตรวจชนิดและช่วงค่าให้อัตโนมัติ ถ้าผิดจะตอบ 422
- `@app.get` / `@app.post` คือ endpoint 3 ตัว: `/health`, `/model-info`, `/predict`

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Diabetes Prediction API")

# อนุญาตให้หน้าเว็บที่อยู่คนละ origin เรียก API ได้ (ใช้งานจริงควรระบุโดเมนแทน "*")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# รูปแบบข้อมูลขาเข้า: Pydantic ตรวจชนิดและช่วงค่าให้อัตโนมัติ (ผิดจะตอบ 422)
class PatientInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., gt=0, le=300)
    BloodPressure: float = Field(..., gt=0, le=200)
    SkinThickness: float = Field(..., gt=0, le=100)
    Insulin: float = Field(..., gt=0, le=900)
    BMI: float = Field(..., gt=0, le=80)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3)
    Age: int = Field(..., ge=1, le=120)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return meta                                    # ชื่อโมเดล + ผลวัด ให้หน้าเว็บแสดง

@app.post("/predict")
def predict(patient: PatientInput):
    X = pd.DataFrame([patient.model_dump()], columns=FEATURES)   # เรียงคอลัมน์ตอน train
    X_s = scaler.transform(X)                                    # scale
    pred = int(knn.predict(X_s)[0])                              # ทำนาย
    prob = float(knn.predict_proba(X_s)[0, 1])
    return {"prediction": pred, "label": meta["classes"][str(pred)], "probability": round(prob, 4)}
```

**3.3 สั่งรัน API ใน thread เบื้องหลัง** `uvicorn.Server` รันแอปใน thread แยก cell จึงจบได้ทันทีแต่เซิร์ฟเวอร์ยังทำงานต่อ แล้วทดสอบเรียกจากใน Colab เอง

```python
import threading, time, requests, uvicorn

def api_is_up():
    try:
        return requests.get("http://localhost:8000/health", timeout=1).ok
    except requests.exceptions.RequestException:
        return False

if api_is_up():
    print("มีเซิร์ฟเวอร์รันอยู่แล้วที่พอร์ต 8000 (ถ้าเพิ่งแก้โค้ด API ให้หยุดก่อนด้วย server.should_exit = True)")
else:
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(60):                          # รอเซิร์ฟเวอร์พร้อม สูงสุด 30 วินาที
        if server.started or not thread.is_alive():
            break
        time.sleep(0.5)

    if not thread.is_alive():
        raise RuntimeError("เซิร์ฟเวอร์หยุดทำงาน ดูข้อความ error ด้านบน")
    if not server.started:
        raise RuntimeError("เซิร์ฟเวอร์ยังไม่พร้อมภายใน 30 วินาที ลองรัน cell นี้อีกครั้ง")

print(requests.get("http://localhost:8000/health").json())
```

cell นี้รอจนเซิร์ฟเวอร์พร้อมจริงก่อนเรียก `/health` (บน Colab อาจใช้เวลาหลายวินาที ถ้าเรียกเร็วเกินจะได้ `ConnectionRefusedError`) ระหว่างรอจะเห็น log ของ uvicorn เช่น `Uvicorn running on http://0.0.0.0:8000` ถ้าเซิร์ฟเวอร์ล้ม error จะแสดงใน log ด้านบน และถ้ามีเซิร์ฟเวอร์รันอยู่แล้ว cell จะไม่เปิดซ้ำ

ถ้าแก้โค้ด API ในข้อ 3.2 ให้หยุดเซิร์ฟเวอร์ก่อนด้วย `server.should_exit = True` (ข้อ 3.7) แล้วรันข้อ 3.2 และ 3.3 ใหม่ URL ของ ngrok ยังใช้ต่อได้เพราะชี้ไปพอร์ต 8000 เหมือนเดิม

ควรได้ `{'status': 'ok'}`

**3.4 เปิด tunnel ด้วย ngrok** อ่าน token จาก Colab Secrets แล้วเปิด tunnel ไปยังพอร์ต 8000

```python
from pyngrok import ngrok

try:
    from google.colab import userdata
    token = userdata.get("NGROK_AUTHTOKEN")       # อ่านจาก Colab Secrets
except Exception:
    from getpass import getpass
    token = getpass("วาง ngrok authtoken: ")

ngrok.set_auth_token(token)
public_url = ngrok.connect(8000, "http").public_url
print("API URL:", public_url)
print("Swagger:", public_url + "/docs")
```

ควรได้ `API URL: https://xxxx.ngrok-free.app` (ตัวอักษรสุ่มต่างกันทุกครั้ง) คัดลอก URL นี้ไปวางในช่อง "API URL" ของหน้าเว็บ `web-api/` แล้วกด "เชื่อมต่อ" หรือเปิดลิงก์ Swagger เพื่อทดลองยิง API จากเบราว์เซอร์

**3.5 ทดสอบเรียกผ่าน URL สาธารณะ** header `ngrok-skip-browser-warning` บอก ngrok แบบฟรีไม่ให้แทรกหน้าคำเตือนแทนข้อมูล JSON ฝั่งหน้าเว็บก็ต้องส่ง header นี้เช่นกัน

```python
r = requests.post(f"{public_url}/predict",
                  headers={"ngrok-skip-browser-warning": "true"},
                  json=patient)                    # patient จากขั้นที่ 2
print(r.json())
```

ควรได้ `{'prediction': 1, 'label': 'เป็นเบาหวาน', 'probability': 0.7333}`

**3.6 ตัวอย่างหน้าเว็บเรียก API** หน้า HTML + JavaScript ไฟล์เดียว ทำ 3 อย่าง: อ่านค่าจากฟอร์มเป็น object → `fetch` แบบ POST ไป `API_URL/predict` พร้อม header ของ ngrok → แสดงผล cell นี้เก็บ HTML ไว้ในตัวแปร `html` แทน `__API_URL__` ด้วย URL ของ ngrok จากข้อ 3.4 แล้วบันทึกเป็น `predict.html` และดาวน์โหลดลงเครื่อง

```python
html = """<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ทำนายเบาหวาน (ตัวอย่างเรียก API)</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 480px; margin: 32px auto; padding: 0 16px; }
  label { display: flex; justify-content: space-between; gap: 12px; margin: 6px 0; }
  input { width: 120px; }
  button { margin-top: 12px; padding: 8px 16px; }
  #result { margin-top: 16px; font-size: 20px; font-weight: bold; }
</style>
</head>
<body>
<h1>ทำนายเบาหวาน</h1>

<form id="form">
  <label>Pregnancies <input name="Pregnancies" type="number" value="6" required></label>
  <label>Glucose <input name="Glucose" type="number" value="148" required></label>
  <label>BloodPressure <input name="BloodPressure" type="number" value="72" required></label>
  <label>SkinThickness <input name="SkinThickness" type="number" value="35" required></label>
  <label>Insulin <input name="Insulin" type="number" value="125" required></label>
  <label>BMI <input name="BMI" type="number" step="0.1" value="33.6" required></label>
  <label>DiabetesPedigreeFunction <input name="DiabetesPedigreeFunction" type="number" step="0.001" value="0.627" required></label>
  <label>Age <input name="Age" type="number" value="50" required></label>
  <button type="submit">ทำนาย</button>
</form>

<div id="result"></div>

<script>
// URL ของ API (ngrok จาก Colab หรือ http://localhost:8000)
const API_URL = "__API_URL__";

document.getElementById("form").addEventListener("submit", async (e) => {
  e.preventDefault();                                   // ไม่ให้ฟอร์มรีโหลดหน้า

  // 1) อ่านค่าจากฟอร์มเป็น object { Pregnancies: 6, Glucose: 148, ... }
  const data = {};
  for (const [name, value] of new FormData(e.target)) data[name] = Number(value);

  // 2) ส่งไปให้ API ทำนาย
  const res = await fetch(API_URL + "/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "true",            // ข้ามหน้าคำเตือนของ ngrok แบบฟรี
    },
    body: JSON.stringify(data),
  });
  const out = await res.json();                         // { prediction, label, probability }

  // 3) แสดงผล
  document.getElementById("result").textContent = res.ok
    ? `${out.label} (ความน่าจะเป็น ${(out.probability * 100).toFixed(1)}%)`
    : "ข้อมูลไม่ถูกต้อง: " + JSON.stringify(out.detail);
});
</script>
</body>
</html>"""

with open("predict.html", "w", encoding="utf-8") as f:
    f.write(html.replace("__API_URL__", public_url))

from google.colab import files
files.download("predict.html")
```

ดับเบิลคลิกเปิด `predict.html` ที่ดาวน์โหลดมา แล้วกด "ทำนาย" ควรขึ้น `เป็นเบาหวาน (ความน่าจะเป็น 73.3%)` ลองแก้ Glucose เป็น 0 จะได้ข้อความ error จาก API เพราะไม่ผ่านการตรวจของ Pydantic หน้าเว็บแบบเต็มที่มีดีไซน์และช่องวาง API URL อยู่ใน `web-api/index.html` ใน zip

**3.7 เลิกใช้งาน** ลบ `#` หน้า 2 บรรทัดแล้วรัน เพื่อปิด tunnel และหยุดเซิร์ฟเวอร์ (หรือปิด Colab ไปเลยก็ได้) cell นี้ถูก comment ไว้ เพื่อไม่ให้ Run all ปิด API ทันทีหลังเปิด ถ้าแค่จะแก้โค้ด API ให้ใช้บรรทัด `server.should_exit = True` บรรทัดเดียว

```python
# ngrok.kill()
# server.should_exit = True
```

API ทำงานเท่าที่ Colab ยังเชื่อมต่ออยู่ ถ้า Colab ตัดการเชื่อมต่อ (ไม่ได้ใช้งานนาน) ต้องรัน notebook 2 ใหม่ตั้งแต่ต้น (อัปโหลดไฟล์ในขั้นที่ 1 ใหม่ด้วย) และ URL ของ ngrok แบบฟรีจะเปลี่ยนใหม่ทุกครั้ง

## แบบ A: FastAPI + JavaScript เรียกผ่าน fetch

เซิร์ฟเวอร์ Python โหลด `diabetes_model.joblib` ครั้งเดียวตอนเริ่ม แล้วเปิด `POST /predict` ให้หน้าเว็บส่ง JSON มา หน้าเว็บเป็น HTML + JavaScript ธรรมดา ไม่มี framework และไม่ต้องใช้ไลบรารีเพิ่ม

รัน API ได้ 2 ทาง ตรรกะเดียวกัน (บน Colab เขียนโค้ดไว้ใน cell ส่วนบนเครื่องใช้ไฟล์ `backend/main.py`)

| ทางเลือก | ทำอย่างไร | URL ที่ได้ |
| --- | --- | --- |
| **Colab + ngrok** (ง่ายสุด) | รัน notebook 2 (06\_diabetes\_backend\_api.ipynb) ถึงขั้นที่ 3 | `https://xxxx.ngrok-free.app` เปลี่ยนทุกครั้งที่รันใหม่ |
| บนเครื่องตัวเอง | `uvicorn main:app --port 8000` ในโฟลเดอร์ `backend/` | `http://localhost:8000` |

### รัน `backend/main.py` บนเครื่องตัวเอง (ทางเลือกที่ 2)

ต้องมี Python 3.10 ขึ้นไปบนเครื่อง เช็กด้วย `python --version` (บน macOS อาจต้องใช้ `python3` แทน `python` ทุกคำสั่ง)

1. แตกไฟล์ zip แล้ววาง `diabetes_model.joblib` และ `model_meta.json` ที่ดาวน์โหลดจาก notebook 1 ขั้นที่ 7 ลงในโฟลเดอร์ `backend/` ทับไฟล์ตัวอย่างเดิม
2. เปิด `model_meta.json` ดูค่า `sklearn_version` แล้วแก้บรรทัด `scikit-learn==` ใน `backend/requirements.txt` ให้เป็นเวอร์ชันเดียวกัน (โมเดลที่บันทึกด้วยเวอร์ชันหนึ่งอาจโหลดด้วยอีกเวอร์ชันไม่ได้)
3. เปิด Terminal (macOS/Linux) หรือ Command Prompt / PowerShell (Windows) เข้าโฟลเดอร์ `backend` แล้วสร้าง virtual environment และติดตั้งไลบรารี

```bash
cd diabetes-deploy/backend

# สร้าง virtual environment (ทำครั้งแรกครั้งเดียว)
python -m venv .venv

# เปิดใช้งาน virtual environment (ทำทุกครั้งที่เปิด terminal ใหม่)
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows

# ติดตั้งไลบรารีจาก requirements.txt (ทำครั้งแรกครั้งเดียว)
pip install -r requirements.txt
```

4. รัน API ด้วย uvicorn (`main` คือไฟล์ `main.py`, `app` คือตัวแปร `app = FastAPI(...)` ในไฟล์)

```bash
uvicorn main:app --reload --port 8000
```

เห็นข้อความ `Uvicorn running on http://127.0.0.1:8000` แปลว่าพร้อมใช้งาน `--reload` ทำให้เซิร์ฟเวอร์รีสตาร์ตเองเมื่อแก้ `main.py` ปล่อย terminal นี้เปิดค้างไว้ และหยุดเซิร์ฟเวอร์ด้วย Ctrl + C

5. ทดสอบ API: เปิดเบราว์เซอร์ไปที่ http://localhost:8000/docs → `POST /predict` → Try it out → Execute ควรได้ `"probability": 0.7333` หรือทดสอบจาก terminal อีกหน้าต่าง

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"Pregnancies\":6,\"Glucose\":148,\"BloodPressure\":72,\"SkinThickness\":35,\"Insulin\":125,\"BMI\":33.6,\"DiabetesPedigreeFunction\":0.627,\"Age\":50}"
```

6. เปิดหน้าเว็บ: ดับเบิลคลิก `web-api/index.html` ใส่ `http://localhost:8000` ในช่อง API URL แล้วกด "เชื่อมต่อ" (หรือใน `predict.html` แก้ `__API_URL__` เป็น `http://localhost:8000`)

**Endpoint ของ API** (เหมือนกันทั้งบน Colab และ `backend/main.py`)

| Method | Path | ใช้ทำอะไร |
| --- | --- | --- |
| GET | `/health` | เช็กว่าเซิร์ฟเวอร์ทำงาน |
| GET | `/model-info` | ส่ง `model_meta.json` ให้หน้าเว็บแสดงชื่อโมเดลและผลประเมิน |
| POST | `/predict` | รับข้อมูล 1 คน คืน `prediction`, `label`, `probability` |
| GET | `/docs` | หน้า Swagger ทดลองยิง API ได้ทันที (FastAPI สร้างให้) |

หัวใจของ backend มี 3 ส่วน:

```python
bundle = joblib.load("diabetes_model.joblib")    # 1) โหลดครั้งเดียว: scaler + knn
scaler, knn = bundle["scaler"], bundle["model"]

class PatientInput(BaseModel):                     # 2) Pydantic ตรวจชนิด/ช่วงค่าให้อัตโนมัติ
    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., gt=0, le=300)      # ต้องกรอกทุกช่อง
    ...

@app.post("/predict")                             # 3) เรียงคอลัมน์ → scale → ทำนาย
def predict(patient: PatientInput):
    X = pd.DataFrame([patient.model_dump()], columns=FEATURES)
    X_s = scaler.transform(X)
    pred = int(knn.predict(X_s)[0])
    prob = float(knn.predict_proba(X_s)[0, 1])
    return {"prediction": pred, "label": meta["classes"][str(pred)], "probability": round(prob, 4)}
```

ฝั่งเว็บ (`web-api/app.js`) ทำ 3 ขั้น: อ่านฟอร์มเป็น object → `fetch` ไป API → เอาผลไปวาด

```javascript
const res = await fetch(`${API_URL}/predict`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),        // { Pregnancies: 6, Glucose: 148, Insulin: 125, ... }
});
const out = await res.json();        // { prediction: 1, label: "เป็นเบาหวาน", probability: 0.7333 }
```

**ช่อง API URL** หน้าเว็บมีช่องให้วาง URL ของ API แล้วกด "เชื่อมต่อ" หน้าเว็บจะเรียก `GET /model-info` เพื่อเช็กว่าเชื่อมต่อได้ และจำ URL ไว้ใช้ครั้งหน้า (`localStorage`) ทุกคำขอส่ง header `ngrok-skip-browser-warning` ไปด้วย เพราะ ngrok แบบฟรีจะแทรกหน้าคำเตือนเป็น HTML แทน JSON ถ้าไม่มี header นี้

```javascript
const API_HEADERS = { "ngrok-skip-browser-warning": "true" };

async function connect() {
  API_URL = ($("api-url").value.trim() || DEFAULT_API_URL).replace(/\/+$/, "");   // ตัด / ท้าย URL
  const res = await fetch(`${API_URL}/model-info`, { headers: API_HEADERS });
  showModelInfo(await res.json());
}
```

เมื่อใช้ Colab + ngrok เปิด `web-api/index.html` ได้ด้วยการดับเบิลคลิกไฟล์เลย ไม่ต้องรัน web server เพราะ URL ของ ngrok เป็น https และ API เปิด CORS ให้ทุก origin

ตัวอย่างผลจริงจากโมเดล KNN: ผู้ป่วยแถวแรกของ dataset (อายุ 50, Glucose 148, BMI 33.6) ได้ 73.3% → เป็นเบาหวาน ส่วนแถวที่สอง (อายุ 31, Glucose 85, BMI 26.6) ได้ 0.0% → ไม่เป็น (KNN ให้ 0% ได้เมื่อเพื่อนบ้าน 15 คนที่ใกล้ที่สุดไม่เป็นเบาหวานเลย) หน้าเว็บดึง `GET /model-info` มาแสดง Accuracy, Precision, Recall, F1-Score ใต้ผลทำนาย (เน้น F1-Score เป็นตัวหลัก)

ต้องเปิด **CORS** ใน FastAPI (`CORSMiddleware`) เพราะหน้าเว็บกับ API อยู่คนละ origin (ต่างโดเมนหรือต่างพอร์ต) และต้องอนุญาต header `ngrok-skip-browser-warning` ด้วย ซึ่ง `allow_headers=["*"]` ครอบคลุมแล้ว ตอนใช้งานจริงให้ระบุโดเมนของเว็บแทน `"*"`

## แบบ B: ONNX + onnxruntime-web ทำนายในเบราว์เซอร์

JavaScript เรียกใช้ไฟล์ `.joblib` ตรงๆ ไม่ได้ เพราะเป็นออบเจกต์ Python ทางออกคือแปลง KNN เป็น **ONNX** (รูปแบบโมเดลกลางที่หลายภาษาอ่านได้) แล้วใช้ไลบรารี `onnxruntime-web` รันในเบราว์เซอร์ ไม่ต้องมี backend และข้อมูลผู้ป่วยไม่ออกจากเครื่อง

**ใน Colab** การแปลงเป็น ONNX และการตรวจผลอยู่ในขั้นที่ 6 ของ Notebook 1

ไฟล์ที่ได้มี input ชื่อ `input` และ output 2 ตัวคือ `label` กับ `probabilities` เมื่อเทียบกับ scikit-learn บนข้อมูลทั้ง 768 แถว label ตรงกัน 100% และความน่าจะเป็นต่างกันไม่เกิน 0.000001 ทดสอบแล้วทั้ง 3 โมเดล: Decision Tree 2.9 KB, Naive Bayes 2.4 KB, KNN 23 KB (KNN ใหญ่กว่าเพราะต้องเก็บข้อมูล train ทั้ง 768 แถวไว้ในไฟล์เพื่อหาเพื่อนบ้าน)

**ในหน้าเว็บ (`web-onnx/`)** โหลดไลบรารีด้วยแท็กเดียว แล้วใช้ตัวแปร global `ort`

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.js"></script>
```

```javascript
const session = await ort.InferenceSession.create("diabetes_model.onnx");   // โหลดครั้งเดียว

// 1) object -> array ตัวเลข เรียงตาม FEATURES
const raw = FEATURES.map(f => data[f]);

// 2) scale แบบเดียวกับ StandardScaler ใน Python: (x - mean) / scale
const { mean, scale } = meta.scaler;                // จาก model_meta.json
const row = raw.map((x, i) => (x - mean[i]) / scale[i]);
const input = new ort.Tensor("float32", Float32Array.from(row), [1, 8]);

// 3) ทำนาย
const out = await session.run({ input });
const prediction  = Number(out.label.data[0]);      // int64 มาเป็น BigInt ต้องแปลง
const probability = out.probabilities.data[1];      // [P(0), P(1)]
```

ให้ผลเท่ากับแบบ A ทุกตัวอย่าง (73.3% และ 0.0%) ลองกดได้ที่หน้าพรีวิว [ประเมินความเสี่ยงเบาหวาน](https://claude.ai/artifact/82nQ6gy42tLaYr2oJru2qE) ซึ่งรันโมเดล KNN นี้จริงในเบราว์เซอร์ และอ่านผลวัด 5 ตัวจาก `model_meta.json`

ข้อควรรู้: หน้าแบบ B ต้องเปิดผ่าน web server (เช่น `python -m http.server 5500`) เพราะเบราว์เซอร์ไม่ให้ `fetch` ไฟล์ `.onnx` จาก `file://`

## รันทดสอบ เลือกแบบไหน และปัญหาที่พบบ่อย

**ข้อมูลตัวอย่างสำหรับทดลอง** 5 คนนี้มาจาก `diabetes_clean.csv` เลือกให้ความน่าจะเป็นไล่ตั้งแต่ไม่เป็นชัดเจนไปจนถึงเป็นชัดเจน กรอกลงหน้าเว็บ (`predict.html`, `web-api/` หรือ `web-onnx/`) หรือ Swagger ที่ `/docs` แล้วผลควรตรงกับคอลัมน์ขวาสุด (โมเดล KNN จากขั้นที่ 4)

| # | Pregnancies | Glucose | BloodPressure | SkinThickness | Insulin | BMI | DiabetesPedigreeFunction | Age | ผลที่ควรได้ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 89 | 66 | 23 | 94 | 28.1 | 0.167 | 21 | ไม่เป็น 0.0% |
| 2 | 5 | 116 | 74 | 29 | 125 | 25.6 | 0.201 | 30 | ไม่เป็น 20.0% |
| 3 | 5 | 109 | 75 | 26 | 125 | 36.0 | 0.546 | 60 | ไม่เป็น 46.7% (เกือบถึงเกณฑ์) |
| 4 | 7 | 147 | 76 | 29 | 125 | 39.4 | 0.257 | 43 | เป็น 60.0% |
| 5 | 8 | 176 | 90 | 34 | 300 | 33.7 | 0.467 | 58 | เป็น 93.3% |

KNN ใช้ 15 เพื่อนบ้าน ความน่าจะเป็นจึงออกมาเป็นขั้นละ 1/15 (6.7%) เช่น 7 ใน 15 คนเป็นเบาหวาน = 46.7% คนที่ 3 ต่ำกว่าเกณฑ์ 50% นิดเดียวจึงยังทายว่าไม่เป็น ลองเพิ่ม Glucose ของคนนี้แล้วดูว่าผลเปลี่ยนเมื่อไร

**รันบนเครื่อง** (โฟลเดอร์ `diabetes-deploy/` ใน zip)

แบบ A ทางที่ 1 (Colab + ngrok): รัน notebook 2 (06\_diabetes\_backend\_api.ipynb) ถึงขั้นที่ 3 → ดับเบิลคลิกเปิด `web-api/index.html` → วาง API URL → กด "เชื่อมต่อ"

แบบ A ทางที่ 2 และแบบ B บนเครื่องตัวเอง:

```bash
# แบบ A: API บนเครื่อง
cd backend
pip install -r requirements.txt          # แก้ scikit-learn== ให้ตรงกับ Colab ก่อน
uvicorn main:app --reload --port 8000    # ทดสอบที่ http://localhost:8000/docs

# หน้าเว็บ (อีกเทอร์มินัล ที่โฟลเดอร์ diabetes-deploy)
python -m http.server 5500
# แบบ A: http://localhost:5500/web-api/   (ช่อง API URL ใช้ http://localhost:8000)
# แบบ B: http://localhost:5500/web-onnx/
```

**เลือกแบบไหน**

|  | แบบ A: API | แบบ B: ONNX ในเบราว์เซอร์ |
| --- | --- | --- |
| ต้องมีเซิร์ฟเวอร์ Python | ใช่ | ไม่ (โฮสต์เป็นเว็บนิ่งได้ เช่น GitHub Pages) |
| ข้อมูลผู้ใช้ | ส่งไปเซิร์ฟเวอร์ | อยู่ในเครื่องผู้ใช้ |
| ใช้โมเดล/โค้ด Python อะไรก็ได้ | ได้ | เฉพาะที่ skl2onnx รองรับ |
| ซ่อนโมเดลจากผู้ใช้ | ได้ | ไม่ได้ (ไฟล์ .onnx ดาวน์โหลดได้) |
| ขนาดที่ผู้ใช้โหลด | หน้าเว็บอย่างเดียว | + ไลบรารี wasm ราว 14 MB (ครั้งแรก) |
| อัปเดตโมเดล | เปลี่ยนไฟล์บนเซิร์ฟเวอร์ | เปลี่ยนไฟล์ .onnx บนเว็บ |

สำหรับงานเรียน/สาธิตแบบ B ง่ายกว่าเพราะไม่ต้องดูแลเซิร์ฟเวอร์ ส่วนระบบจริงที่มีข้อมูลอ่อนไหวหรือโมเดลซับซ้อนใช้แบบ A

**ปัญหาที่พบบ่อย**

| อาการ | สาเหตุ | วิธีแก้ |
| --- | --- | --- |
| `InconsistentVersionWarning` หรือโหลด .joblib พัง | เวอร์ชัน scikit-learn ไม่ตรงกับ Colab | ติดตั้งเวอร์ชันเดียวกับที่ขั้นที่ 1 พิมพ์ (หรือใช้ Colab + ngrok) |
| Colab หาไฟล์ `diabetes_clean.csv` ไม่เจอ | session ใหม่ ไฟล์ที่อัปโหลดหายไป | รันขั้นที่ 2 แล้วอัปโหลดไฟล์ใหม่ |
| notebook 2 ข้อ 3.3 ได้ `ConnectionRefusedError` | เรียก `/health` ก่อนเซิร์ฟเวอร์พร้อม หรือเซิร์ฟเวอร์ล้ม | ใช้ cell 3.3 เวอร์ชันที่รอจนพร้อม และอ่าน error ใน log ของ uvicorn |
| notebook 2 ข้อ 3.3 แจ้ง address already in use | มีเซิร์ฟเวอร์เก่ารันอยู่ที่พอร์ต 8000 | `server.should_exit = True` แล้วรันใหม่ หรือ Runtime → Restart session |
| notebook 2 ข้อ 3.4 error เรื่อง authtoken | ไม่ได้ตั้ง Secret หรือไม่ได้เปิด Notebook access | ตั้ง `NGROK_AUTHTOKEN` ใน Secrets แล้วเปิดสวิตช์ หรือวาง token เมื่อ cell ถาม |
| หน้าเว็บขึ้น "เชื่อมต่อ API ไม่ได้" | Colab หลุด, URL ngrok เปลี่ยน, หรือเซิร์ฟเวอร์หยุด | รัน notebook 2 ใหม่แล้ววาง URL ใหม่ |
| API ตอบ 422 | เว้นว่างบางช่อง หรือค่าเกินช่วงที่กำหนดใน Pydantic | ดูข้อความ error ที่หน้าเว็บแสดง |
| แบบ B ขึ้น "โหลดโมเดลไม่สำเร็จ" | เปิดไฟล์ด้วย `file://` | เปิดผ่าน `python -m http.server` |
| ผลแบบ A กับ B ไม่เท่ากัน | ลำดับ feature ใน JS ไม่ตรงกับตอน train | ใช้ `FEATURES` ชุดเดียวกับ `model_meta.json` |

**ต่อยอด**: ปรับ `class_weight` หรือ threshold ให้ recall สูงขึ้นสำหรับงานคัดกรอง, ใช้ `GridSearchCV` จูนพารามิเตอร์, deploy API บน Render/Railway แล้วชี้ `API_URL` ไปที่นั่น, หรือวางแบบ B บน GitHub Pages

ตัวอย่างนี้เพื่อการเรียนรู้เท่านั้น ไม่ใช่การวินิจฉัยทางการแพทย์
