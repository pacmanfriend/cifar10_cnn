# CIFAR-10 CNN

Двухслойная свёрточная нейросеть для классификации CIFAR-10, написанная на Rust.  
CPU-путь использует собственный граф вычислений; GPU-путь — CUDA-ядра через NVRTC.  
Управление через браузерный UI, бэкенд — HTTP API на Axum.

## Требования

Для Docker-запуска нужны Docker и, если используется GPU, NVIDIA Container Toolkit с доступом к `--gpus all`.
Для локальной разработки используются версии, закреплённые в проекте:

- Rust `1.83.0` через `rust-toolchain.toml`
- Node.js `22.x` (`Dockerfile` использует `22.15.0`)
- CUDA `12.8` для GPU-бэкенда

CPU-режим можно запускать без CUDA-совместимой видеокарты.

## Быстрый старт

### 1. Собрать образ

```bash
docker build -t cifar10-cnn .
```

Сборка займёт несколько минут: компилируются Rust-бинарь и SvelteKit UI.

### 2. Запустить контейнер

**С GPU (рекомендуется):**
```bash
docker run --rm --gpus all -p 8080:8080 cifar10-cnn
```

**Только CPU:**
```bash
docker run --rm -p 8080:8080 cifar10-cnn
```

### 3. Открыть UI

Перейдите в браузере по адресу:

```
http://localhost:8080
```

Сервер отдаёт UI на корневом пути и API по префиксу `/api`.

---

## Локальная разработка

### Production-like запуск из репозитория

Сначала соберите статический SvelteKit UI:

```bash
cd web
npm ci
npm run build
cd ..
```

Затем запустите Rust-сервер из корня репозитория:

```bash
cargo run -- --port 8080
```

Сервер будет раздавать `web/build/` и API на `http://localhost:8080`.

### Dev-режим UI

В одном терминале запустите API:

```bash
cargo run -- --port 8080
```

Во втором терминале запустите Vite dev server:

```bash
cd web
npm run dev
```

Vite проксирует запросы `/api` на `http://localhost:8080`.

---

## Обучение на CIFAR-10

Для запуска обучения нужен датасет в бинарном формате.

### Скачать датасет

```bash
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
# Появится папка cifar-10-batches-bin/
```

### Запустить контейнер с данными

```bash
docker run --rm --gpus all \
  -p 8080:8080 \
  -v /путь/до/cifar-10-batches-bin:/app/data/cifar-10-batches-bin:ro \
  cifar10-cnn
```

Папка монтируется в `/app/data/cifar-10-batches-bin` внутри контейнера — это путь, который UI подставляет по умолчанию в форме обучения.

### Запустить обучение через UI

1. Откройте `http://localhost:8080/train`
2. Поле **Data directory** оставьте как есть: `data/cifar-10-batches-bin`
3. Нажмите **Start** — обучение запустится в фоне
4. График loss и accuracy обновляется автоматически раз в секунду

---

## Страницы UI

| Путь | Что делает |
|------|-----------|
| `/` | Дашборд: статус CPU/GPU и модели |
| `/system` | Детальная информация о CPU и GPU |
| `/model` | Загрузка/сохранение весов, переключение CPU↔GPU |
| `/train` | Запуск обучения, график прогресса |
| `/predict` | Загрузка изображения и предсказание класса |

---

## Сохранение и загрузка весов

На странице `/model`:

- **Save** — укажите путь внутри контейнера, например `weights/model.ck10`, и нажмите Save
- **Load** — укажите тот же путь и нажмите Load

Чтобы веса сохранялись между перезапусками контейнера, примонтируйте папку:

```bash
docker run --rm --gpus all \
  -p 8080:8080 \
  -v /путь/до/cifar-10-batches-bin:/app/data/cifar-10-batches-bin:ro \
  -v /путь/для/весов:/app/weights \
  cifar10-cnn
```

Тогда в UI указывайте путь `weights/model.ck10`.

Файл `.ck10` — собственный бинарный checkpoint-формат проекта, не PyTorch/ONNX.

---

## Проверки

Перед фиксацией изменений полезно запускать:

```bash
cargo fmt --check
cargo test
cargo clippy -- -D warnings
```

Для фронтенда:

```bash
cd web
npm run check
npm run build
```

---

## Параметры запуска

Единственный аргумент бинаря — порт:

```bash
docker run --rm -p 9090:9090 cifar10-cnn ./cifar10_cnn --port 9090
```

По умолчанию используется порт `8080`.
