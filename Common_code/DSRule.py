# ───────── DSRule.py ─────────
class DSRule:
    """
    Обёртка над λ-предикатом rule.ld.
    usability рассчитывается ТОЛЬКО на тестовой выборке
    и задаётся после обучения (см. test_Ripper_DST.py).
    """
    def __init__(self, ld, caption: str = ""):
        self.ld = ld                  # сам предикат (lambda)
        self.caption = caption        # текстовое описание условия
        self.freq = 0         # number of training instances this rule covers
        self.usability: float | None = None   # % тестовых примеров, покрытых правилом

    # красивый вывод: если usability ещё нет – без него
    def __str__(self):
        if self.usability is None:
            return self.caption
        return f"{self.caption} | usability={self.usability:.1f}%"

    # чтобы правило можно было звать как функцию
    __call__ = lambda self, *a, **kw: self.ld(*a, **kw)
