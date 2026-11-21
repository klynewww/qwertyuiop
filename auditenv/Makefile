PYTHON ?= python
EPISODES = data

.PHONY: test check audit episodes judge demo replay docker-build docker-check clean-runs

# быстрые тесты
test:
	$(PYTHON) -m pytest -q

# 20 открытых эпизодов (10 train + 10 dev) генератором A
episodes:
	$(PYTHON) -m generator.generate_v1 --out $(EPISODES) --split train
	$(PYTHON) -m generator.generate_v1 --out $(EPISODES) --split dev

# самооценка по пяти критериям жюри: баллы из 100
audit:
	$(PYTHON) -m scripts.audit

# тесты + все политики на 20 эпизодах, агент в отдельном процессе; без сети и GPU
check: test episodes
	$(PYTHON) -m scripts.run_policies --episodes-dir $(EPISODES) --episodes 20 --isolated
	$(PYTHON) -m scripts.run_judge --out runs/judge

# организаторские политики через judge_adapter на матрице D01-D20 (так проверяет жюри)
judge:
	$(PYTHON) -m scripts.run_judge --out runs/judge

# пошаговый показ: дотошный аудитор, «пометить всё» и попытка прочитать эталон
demo:
	$(PYTHON) -m scripts.demo

# HTML-лента аудита (docs/replay.html)
replay:
	$(PYTHON) -m scripts.make_replay

docker-build:
	docker build -t auditenv .

# то же самое (проверка + самооценка), но в контейнере без сети
docker-check: docker-build
	docker run --rm --network none auditenv

clean-runs:
	rm -rf runs
