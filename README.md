# HW8. Мониторинг и наблюдаемость ML-системы

## 1. Дерево метрик ML-системы

### Бизнес-метрики
- Revenue uplift от персонализированной рекламы.
- CTR / conversion rate по встроенным брендам.
- Ad fill rate: доля видеопоказов, где удалось корректно встроить бренд.
- Жалобы пользователей / legal incidents.
- Retention и watch time после вставки рекламы.

### Метрики приложения
- Availability ML API > 99%.
- p95 latency < 1 сек.
- Error rate < 1%.
- RPS / throughput.
- Доля успешно обработанных кадров.
- Очередь необработанных событий в stream layer.

### ML-метрики
- Accuracy / F1 для детекции объектов и зон вставки.
- Drift score по входным признакам/кадрам.
- Model confidence для вставки бренда.
- Доля отклонений safety-фильтром.
- Деградация качества: падение accuracy/F1 относительно reference-батча.
- Доля hallucination/incorrect placement.

### Метрики инфраструктуры
- CPU/RAM/GPU utilization.
- GPU memory usage.
- Disk usage.
- Network I/O.
- Kafka consumer lag.
- Uptime контейнеров.
- Prometheus target UP/DOWN.

### SLO для демонстрации
- **Latency SLO:** p95 request latency < 1 сек.
- **Availability SLO:** сервис доступен > 99%.
- **Error Rate SLO:** ошибок < 1%.
- **ML Quality SLO:** drift score <= 1.
- **Data Quality SLO:** число DQ-инцидентов = 0.

---

## 2. Запуск Prometheus + Grafana + ML-сервис

- ML-сервис: `http://localhost:8000`
- Метрики: `http://localhost:8000/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Логин Grafana: `admin`
- Пароль Grafana: `admin`

```bash
for i in {1..30}; do curl -s "http://localhost:8000/predict?slow=true" > /dev/null; done
```

Сработат алерт, потому что p95 latency выше 1 секунды.

## 3. MLflow

В сервисе MLflow используется для логирования:

- типа модели;
- версии приложения;
- latency;
- класса предсказания;
- drift score;
- Data Quality incident count.

Логи сохраняются в папку:

```text
mlruns/
```

---

## 4. Обнаружение дрифта и деградации модели

Запуск:

```bash
pip install -r requirements.txt
python drift/evidently_drift_demo.py
```

Скрипт делает две вещи:

1. **Дрифт данных:** берет reference batch и current batch, затем намеренно изменяет current batch:
   `current = current * 3 + 10`.
2. **Деградация модели:** обученная модель получает смещенные данные, из-за чего качество падает.


## 5. Data Quality Ops

### Подготовка PostgreSQL

PostgreSQL поднимается вместе с `docker compose up --build`.
Таблица создается автоматически из файла:

```text
dqops/01_init.sql
```

Подключение:

- host: `localhost`
- port: `5432`
- database: `hw8dq`
- user: `hw8`
- password: `hw8`
- schema: `public`
- table: `product_placement_events`

### Запуск DQOps

```bash
pip install --user dqops
python -m dqops
```

## 6. Архитектура Virtual Product Placement

Выбрана **Kappa-архитектура**, потому что задача потоковая:

- входные данные — видеопоток и поток контекста зрителя;
- нужно обрабатывать кадры почти в реальном времени;
- кадры хорошо параллелятся;
- при ошибках можно переиграть события из Kafka;
- не нужно держать две разные логики batch layer и speed layer, как в Lambda.

Схема:

```text
architecture/virtual_product_placement_kappa.png
```

### Главные стримы

- `raw_video_frames` — поток кадров.
- `viewer_context_stream` — страна, сегмент, контекст зрителя.
- `placement_events_stream` — события успешных вставок бренда.
- `audit_and_feedback_stream` — аудит, обратная связь, ошибки, legal/safety signals.
