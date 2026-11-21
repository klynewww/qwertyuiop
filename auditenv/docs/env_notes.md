# Заметки участника B: среда и oracle

Это не contract.md, а предложения B для обсуждения в блоке 1. ТЗ у B не было, поэтому всё, что помечено «?», надо сверить с контрактом.

## Допущения, которые нужно сверить
- ? Название типа ошибки `DUPLICATE_INVOICE`, поля findings (`type`, `affected_ids`, `amount_minor`, `currency`, `explanation`) и truth (`evidence_sets`).
- ? Колонки ledger/fx и поля invoices/payments (см. tests/fixtures/mini/open).
- ? Документы лежат в `open/docs/<doc_id>.txt`, doc_id = имя файла без расширения.
- ? Id записей уникальны в пределах таблицы; `get_record` принимает `table` и `id`.

## Решения по среде
- Любой вызов step() тратит шаг, включая ошибочный (неизвестная таблица или id, плохой курсор). Иначе можно перебирать id бесплатно.
- Лимит 100 действий включает submit. Если на 100-м действии не submit, эпизод усечён (truncated), хост считает награду как за пустой ответ.
- inspection_steps = число действий-чтений (list_records, get_record, get_document), submit не считается.
- Курсор: `c<смещение>` по записям, отсортированным по id. Порядок не зависит от файла, поэтому стабилен.
- info содержит только `step` и `steps_left`.
- Ошибка: `{"error": {"code": ..., "message": ...}}`. Коды: INVALID_ACTION, UNKNOWN_ACTION, UNKNOWN_TABLE, UNKNOWN_ID, UNKNOWN_DOCUMENT, INVALID_CURSOR, INVALID_ARGUMENT, NOT_IMPLEMENTED (калькулятор завтра).
- `reset` создаёт новый объект состояния эпизода целиком, ничего не переносится.

## Изоляция
- Загрузчик получает путь только к `open/` и читает белый список файлов.
- `AuditEnv` не хранит truth и оценщика; награду считает `HostEpisode` (хост-сторона).
- Oracle получает ответ от хоста (`hidden/findings.json`), а не от среды.
- Ограничение: пока всё в одном процессе, это изоляция на уровне кода и тестов (tests/test_isolation.py). Изоляция на уровне процесса/контейнера: Dockerfile, суббота.

## Как подключить генератор A и судью C
- Генератор кладёт эпизод в `<episode>/open/` и `<episode>/hidden/`, как в tests/fixtures/mini.
- Судья реализует `Grader.grade(findings, inspection_steps) -> float` и заменяет `StubGrader` из tests/helpers.py.


---
# Обновление B (пятница-суббота)

## Что добавлено
- `env/calculator.py`: калькулятор. Разбор через `ast` с белым списком, вычисления в `Decimal`, без `eval`. Разрешены числа, `+ - * /`, унарные знаки, скобки. Остальное даёт `CALC_ERROR` с короткой фразой и тратит шаг.
- Лимит времени 60 с на эпизод (`time_limit_s` в config): по таймауту `truncated=True`, ответ считается пустым.
- `env/harness.py`: `TrajectoryLogger` пишет `trajectories.jsonl` (episode_id, step, action, observation, reward, terminated, truncated), в наблюдениях только открытые данные.
- `env/isolated.py` + `env/agent_runner.py`: агент в отдельном процессе, связь только строками JSON. Песочница в процессе агента запрещает чтение `hidden/`/`private/`, запуск процессов, сеть, ctypes. Завис, упал или нарушил протокол: эпизод усечён, ответ пустой.
- `policies/`: `random` (фиксированный seed), `reconcile` (простая сверка), `flag_all` (пометить всё). Oracle остался хостовым.
- `scripts/run_policies.py`: прогон политик, таблица «политика × профиль», журнал в `runs/`.
- `env/layout.py`: понимает и `public/`+`private/` (генератор A), и `open/`+`hidden/` (мини-эпизод).
- `env/compat.py`: ВРЕМЕННЫЙ переводчик truth генератора A в формат судьи C (см. ниже).
- Тесты: 575 (калькулятор, лимиты и таймаут, журнал, изоляция процессов с «взломщиками», фаззинг 300 эпизодов мусорных действий, данные настоящего генератора).
- `Makefile` (`make test`, `make check`, `make docker-check`), `Dockerfile` (Python 3.11, `--network none`), `requirements.txt`, `LICENSE` (MIT).

## Несовпадения контракта, найденные при подключении генератора A (решить командой)
1. Папки: A пишет `public/`+`private/`, судья C читает `open/`+`hidden/`. Среда понимает оба.
2. Формат truth: A пишет `issues[]` с `expected_amount_minor`, `acceptable_evidence_sets`, `amount_tolerance_minor`; судья C ждёт `errors[]` с `amount_minor`, `evidence_sets`. Сейчас их стыкует `env/compat.py`. Надо выбрать один формат и убрать прослойку.
3. Допуск суммы: у A он у каждой ошибки в truth (сейчас 0), у C в `open/policy.json: amount_tolerance_minor`, а у A в policy.json такого поля нет.
4. Типы ошибок: у A строчные (`duplicate_invoice`), у политик B `DUPLICATE_INVOICE`. Судья сравнивает без учёта регистра, поэтому работает, но имена надо зафиксировать.
5. `findings.json`: у A его нет. Oracle строит ответ из truth: `affected_ids` + один полный набор доказательств (иначе по условию 5 судьи вывод не засчитается). Проверить, что так задумано в ТЗ.
6. Документы: у A отдельных файлов-документов нет, `get_document("DOC-INV-000123")` возвращает саму запись счёта или платежа как текст JSON.
7. ID курсов: колонки id нет, `get_record(table="fx", id="FX:2026-01-05:EUR:USD")` ищет по вычисленному ID из policy.json.

## Допущения B (подтвердить)
- «Два режима» = в одном процессе (быстро, для тестов) и агент в отдельном процессе (для проверки изоляции). Если в ТЗ имелось в виду другое, скажите.
- Калькулятор тратит шаг, но не считается за `inspection_steps` (штраф 0.002 только за чтения).
- Страница `list_records` до 50 записей: чтение всего леджера большого эпизода съедает много из 100 шагов.
- Изоляция по процессам защищает на уровне Python (audit hook). Настоящую границу даёт Docker.

## Не проверено
- Docker: в моём окружении нет демона, `docker build` не запускался. Нужно один раз выполнить `make docker-check` на чистой машине.


---
# Обновление: приведено к определениям жюри (21 ноября)
- Findings: `evidence_ids`, `claimed_amount_minor`; имена типов точные: `duplicate_invoice`, `missing_or_partial_allocation`, `wrong_date_or_fx_rate`, `period_cutoff`.
- `inspection_steps` = все вызовы step до submit, включая ошибочные и калькулятор (раньше считались только чтения).
- `get_document` возвращает объект документа; `limit` больше 50 урезается до 50; ledger отдаётся типизированным (суммы целыми, `null` вместо строки "null").
- Протокол жюри (конверт `{"name","args"}`, наблюдения `{"result","error","remaining_steps"}`) реализован в `judge_adapter.py`; внутренний формат сохранён для наших политик (`policies/judge_shim.py`).
