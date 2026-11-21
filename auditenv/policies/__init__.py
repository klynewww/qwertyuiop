"""Политики. Адаптер = класс с act(observation) -> action (см. common.py).

SPECS: имя -> 'модуль:класс' для запуска в отдельном процессе.
Oracle здесь нет: ему нужен верный ответ, он работает только на стороне хоста.
"""
SPECS = {
    "careful": "policies.careful:CarefulPolicy",
    "careful_dup": "policies.careful:CarefulDupOnly",
    "random": "policies.random_explorer:RandomExplorer",
    "reconcile": "policies.reconcile:ReconcilePolicy",
    "flag_all": "policies.flag_all:FlagAllPolicy",
}
