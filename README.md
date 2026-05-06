# Intent Classification: Few-Shot & Zero-Shot vs Fine-Tuning

Дипломная работа: сравнительный анализ подходов к классификации интентов на датасете CLINC150.

---

## Постановка задачи

Классификация интентов (intent classification) - ключевая задача в диалоговых системах и голосовых ассистентах. Модель должна определить намерение пользователя по его текстовому запросу.

**Датасет:** [CLINC150](https://github.com/clinc/oos-eval) — 150 классов интентов, 22 500 примеров, включая категорию out-of-scope.

### Исследовательские вопросы

1. Насколько Zero-Shot уступает Fine-Tuning на простых и сложных интентах?
2. Сколько примеров нужно Few-Shot, чтобы догнать Fine-Tuning?
3. Какой подход оптимален с учётом затрат на разметку данных?

---

## Гипотеза

> Для простых интентов Zero-Shot будет достаточен, а для сложных и семантически близких — Few-Shot догонит Fine-Tuning уже на 5 примерах.

---

## Стек

- Python 3.10+
- HuggingFace Transformers (DistilBERT, Flan-T5)
- scikit-learn
- Jupyter Notebook

---

## Установка и первый запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/xibrune/intent-classification.git
cd intent-classification
```

### 2. Создать виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

### 4. Запустить Jupyter

```bash
jupyter notebook
```
