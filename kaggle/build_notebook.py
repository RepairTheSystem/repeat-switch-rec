"""
Собирает самодостаточный ноутбук для Kaggle/Colab: исходники проекта вставляются в ячейки %%writefile,
так что загружать на Kaggle нужно только сам .ipynb.

  python kaggle/build_notebook.py                     # → kaggle/repeat_switch_kaggle.ipynb
  python kaggle/build_notebook.py --quick --no-gpu --out <path>   # дымовая версия для локальной проверки

Пересобирай ноутбук после любых правок в .py — иначе на Kaggle уедет старый код.
"""
import argparse
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ["data.py", "metrics.py", "mixing.py", "baselines.py", "heuristics.py", "model.py",
           "train.py", "lr_probe.py", "latency_bench.py"]

INTRO = """# Одностадийная модель с режимом повторов — полный прогон на GPU

Ноутбук самодостаточный: код проекта записывается в файлы ячейками `%%writefile` ниже.

**Как запустить на Kaggle**

1. *Create → New Notebook → File → Import Notebook* — загрузи этот `.ipynb`.
2. Справа *Settings → Accelerator → GPU T4 x2* (или *P100*). Интернет не нужен.
3. Запуск:
   * быстро проверить — поставь `QUICK = True` в ячейке настроек и *Run All* (пара минут);
   * полный прогон — `QUICK = False`, затем *Save Version → Save & Run All (Commit)*.
     Так ноутбук считается в фоне (до 12 ч), вкладку можно закрыть, результаты останутся во вкладке *Output*.
4. Результаты: `results.zip` в *Output* (внутри `summary.md`, `metrics.csv`, `bootstrap.csv`, `latency*.csv`, логи).

Время на T4/P100: прошлый прогон (base + 1 switch) — ~1 мин обучения на прогон; с абляцией (base + 6 switch) —
оценочно ~3–5 мин на прогон, ≈ 30–45 мин на 9 прогонов.

**Colab:** тоже работает — *Runtime → Change runtime type → GPU*, результаты скачай до отключения сессии.
"""

SETUP = """import os, sys, time, torch

REQUIRE_GPU = {require_gpu}
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет GPU")
if REQUIRE_GPU:
    assert torch.cuda.is_available(), "Включи GPU: Settings → Accelerator (иначе прогон займёт часы)"

# рабочая папка: на Kaggle — /kaggle/working, иначе — рядом с ноутбуком
BASE = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
WORK = os.path.join(BASE, "repeat_switch")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
print("работаем в", WORK)
try:
    import tabulate  # noqa: F401  — красивые таблицы в summary.md (без него будет обычный текст)
except ImportError:
    os.system(f"{{sys.executable}} -m pip install -q tabulate")

import subprocess


def run(args, log):
    \"\"\"Запускает python-скрипт, печатает вывод по ходу и пишет его в лог (как `| tee`, но кроссплатформенно).\"\"\"
    print("$ python", args, flush=True)
    env = {{**os.environ, "PYTHONIOENCODING": "utf-8"}}
    with open(log, "w", encoding="utf-8") as f, subprocess.Popen(
            f'"{{sys.executable}}" -u {{args}}', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", env=env) as p:
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
    assert p.returncode == 0, f"процесс упал (код {{p.returncode}}), см. {{log}}"
"""

CONFIG = """# ---- настройки прогона ----
QUICK = {quick}                 # True — дымовой прогон за пару минут, False — полный
SEEDS = "0 1 2"
REGIMES = "high mid low"        # доля повторов ≈ 0.82–0.88 / 0.23–0.54 / 0.13–0.37 (зависит от сида)
COMMON = "--lr-base 1e-3 --lr-switch 3e-3"   # lr подобраны на val (lr_probe.py)
# абляция: switch — исходный; mask — explore исключает всю историю; bias/null/prior — опции repeat-головы
SWITCH_VARIANTS = "switch mask mask+bias mask+null mask+prior mask+all"
RUN_LEAK_CHECK = False         # цена утечки уже измерена (≈ 0); True — пересчитать
# batch оставлен 128, как на CPU: больший батч меняет оптимизацию и lr пришлось бы подбирать заново

if QUICK:
    SEEDS, COMMON = "0", COMMON + " --n-users 500 --max-epochs 2"
os.makedirs("results", exist_ok=True)
print(f"QUICK={{QUICK}} SEEDS={{SEEDS}} REGIMES={{REGIMES}} COMMON={{COMMON}}")
print(f"SWITCH_VARIANTS={{SWITCH_VARIANTS}}")
"""

RUN_MAIN = """# Основной эксперимент: протокол window, все режимы × сиды
t0 = time.time()
run(f"train.py --regimes {REGIMES} --seeds {SEEDS} {COMMON} --switch-variants {SWITCH_VARIANTS} --out results/main",
    "results/main.log")
print(f"основной прогон: {(time.time() - t0) / 60:.1f} мин")
"""

RUN_LEAK = """# Цена утечки времени запроса: старый протокол (время запроса = время покупки) vs та же цель без утечки
if RUN_LEAK_CHECK:
    t0 = time.time()
    for proto, out in (("next", "leak_next"), ("next-start", "leak_next_start")):
        run(f"train.py --protocol {proto} --regimes high --seeds {SEEDS} {COMMON} --switch-variants switch "
            f"--out results/{out}", f"results/{out}.log")
    print(f"проверка утечки: {(time.time() - t0) / 60:.1f} мин")
else:
    print("пропущено (RUN_LEAK_CHECK = False)")
"""

SHOW = """from IPython.display import Markdown, display
import pandas as pd

display(Markdown(open("results/main/summary.md", encoding="utf-8").read()))

# сводка по утечке: Recall@10 на тесте, среднее по сидам
rows = []
for proto in ("next", "next_start"):
    p = f"results/leak_{proto}/metrics.csv"
    if os.path.exists(p):
        m = pd.read_csv(p)
        m = m[(m.split == "test") & m.variant.isin(["freq+pop", "heur/switch", "nn/base", "nn/switch"])]
        rows.append(m.groupby("variant")["Recall@10"].mean().rename(proto))
if rows:
    leak = pd.concat(rows, axis=1)
    leak["цена утечки"] = leak["next"] - leak["next_start"]
    display(leak.round(4))
"""

LATENCY = """# Латентность на чистой машине: архитектура и путь выдачи, без обучения (веса на задержку не влияют)
n = 50 if QUICK else 300
run(f"latency_bench.py --device cpu --n {n} --out results/latency_cpu.csv", "results/latency_cpu.log")
if torch.cuda.is_available():
    run(f"latency_bench.py --device cuda --n {n} --out results/latency_cuda.csv", "results/latency_cuda.log")
for f in ("results/latency_cpu.csv", "results/latency_cuda.csv"):
    if os.path.exists(f):
        display(pd.read_csv(f).round(3))
"""

ZIP = """import shutil
archive = shutil.make_archive(os.path.join(BASE, "results"), "zip", "results")
print("сохранено:", archive)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "kaggle", "repeat_switch_kaggle.ipynb"))
    ap.add_argument("--quick", action="store_true", help="QUICK=True по умолчанию (для проверки)")
    ap.add_argument("--no-gpu", action="store_true", help="не требовать GPU (для локальной проверки)")
    a = ap.parse_args()

    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO),
             nbf.v4.new_code_cell(SETUP.format(require_gpu=not a.no_gpu)),
             nbf.v4.new_markdown_cell("## Код проекта\nСгенерировано `kaggle/build_notebook.py` из исходников.")]
    for name in SOURCES:
        with open(os.path.join(ROOT, name), encoding="utf-8") as f:
            cells.append(nbf.v4.new_code_cell(f"%%writefile {name}\n" + f.read()))
    cells += [nbf.v4.new_markdown_cell("## Прогон"),
              nbf.v4.new_code_cell(CONFIG.format(quick=a.quick)),
              nbf.v4.new_code_cell(RUN_MAIN),
              nbf.v4.new_code_cell(RUN_LEAK),
              nbf.v4.new_markdown_cell("## Результаты"),
              nbf.v4.new_code_cell(SHOW),
              nbf.v4.new_code_cell(LATENCY),
              nbf.v4.new_code_cell(ZIP)]
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"},
                      "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": False}}
    nbf.validate(nb)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    nbf.write(nb, a.out)
    print("written", a.out)


if __name__ == "__main__":
    main()
