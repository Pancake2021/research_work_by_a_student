# Colab Quickstart (GPU/TPU)

## 1) Первая ячейка в Google Colab
```python
!pip -q install flask pyngrok psutil requests
!git clone https://github.com/Pancake2021/research_work_by_a_student.git /content/research_work_by_a_student || true

import os
from google.colab import drive

drive.mount('/content/drive')  # опционально

# Если хочешь закрытый доступ к серверу:
os.environ['COLAB_MCP_API_KEY'] = 'set-your-secret-key'
# Для публичного туннеля ngrok:
os.environ['NGROK_AUTHTOKEN'] = 'set-your-ngrok-token'
os.environ['COLAB_MCP_ENABLE_NGROK'] = '1'

!python /content/research_work_by_a_student/agent/colab_mcp_server.py
```

После старта сервер напечатает строку вида `COLAB_MCP_URL=https://xxxx.ngrok-free.app`.

## 2) Локально на твоём Mac
```bash
cd /Users/glebpankeev/Downloads/research_work_by_a_student-develop
python -m agent.experiment_runner \
  --colab-url "https://xxxx.ngrok-free.app" \
  --colab-api-key "set-your-secret-key" \
  --local-root "."
```

## 3) Что будет создано локально
- `experiment_db.json`
- `runs/<run_id>/stdout.log`
- `runs/<run_id>/stderr.log`
- `runs/<run_id>/metrics_stream.jsonl`

## 4) Строгий порядок экспериментов
Оркестратор запускает цепочку автоматически:
`EXP-01 baseline -> EXP-02 grpo/accuracy -> EXP-03 grpo/reasoning -> EXP-04 grpo/binary -> EXP-05 ppo/reasoning -> EXP-06 dapo/entropy -> EXP-07 lambda_grpo`
