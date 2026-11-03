"""The Galaxy Zoo challenge CSV's ordered 37 regression targets."""

TASK_SIZES = (3, 2, 2, 2, 4, 2, 3, 7, 3, 3, 6)
TASK_NAMES = (
    "Smooth", "EdgeOn", "Bar", "Spiral", "Bulge", "Odd", "Round",
    "OddType", "BulgeShape", "ArmWind", "ArmNum",
)
LABEL_COLUMNS = tuple(
    f"Class{task}.{answer}"
    for task, count in enumerate(TASK_SIZES, start=1)
    for answer in range(1, count + 1)
)
NUM_TARGETS = sum(TASK_SIZES)


def task_slices():
    start = 0
    for index, (name, size) in enumerate(zip(TASK_NAMES, TASK_SIZES), start=1):
        yield f"T{index}:{name}", slice(start, start + size)
        start += size

