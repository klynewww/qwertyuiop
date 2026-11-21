# Судья: соглашения (приведены к определениям жюри от 20.11.2026)

- Структура: `type, affected_ids, evidence_ids, claimed_amount_minor, currency, explanation` обязательны.
  Лимиты: 30 выводов, 10 ID в `affected_ids`, 10 ID в `evidence_ids` (считаются до удаления повторов),
  explanation ≤ 1000 символов Unicode. `claimed_amount_minor` целое, не bool/строка/float.
  Нарушение = неверный submit, награда −1, метрики null, ошибка в журнале оценщика.
- Сопоставление (TP): `type` и `currency` точно; множество `affected_ids` равно каноническому;
  |claimed − expected| ≤ τ (τ из `amount_tolerance_minor` ошибки, по умолчанию 0); `evidence_ids`
  содержат один допустимый набор и все существуют в открытых данных (записи, документы, FX-ID).
  Неизвестный type: FP, не авария.
- Паросочетание: максимальное; при равной мощности: ошибки по `issue_id`, каждой наименьший индекс вывода.
- Метрики: P, R, F1; пустой эталон и пустые выводы: P=R=0, F1=1 (специальное правило).
- Награда: `clip(2·F1 − 1 − 0.002·inspection_steps, −1, 1)`.
- Код: `evaluator/` (schema, normalize, matcher, metrics, grader), `judge_adapter.score_episode`.
