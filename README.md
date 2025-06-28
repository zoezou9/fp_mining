## Using Machine Learning in Frequent Pattern Mining on Data Streams

### 前置需求

- Python 3.x
- virtualenv（或 Conda）

---

## 使用說明

1. **更新程式碼**

   ```bash
   git pull
   ```

2. **建立虛擬環境並啟動**

```
python3 -m venv venv
source venv/bin/activate
```

3. 更新套件

```
pip install --upgrade pip
pip install -r requirements.txt
```

4. **執行並收集 log**

```
/usr/bin/time -v python3 fp_train.py 2>&1 | tee run.log
/usr/bin/time -v python3 -u fp_train.py 2>&1 | tee run.log
```

see summary

```
jq . results_summary.json
head -n 10 frequent_patterns.csv
```

離開 venv

```
deactivate
```
