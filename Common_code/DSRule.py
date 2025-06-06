# DSRule.py
class DSRule(object):
    """
    Wrapper for labeled lambdas, used to print rules in DSModel.
    """
    def __init__(self, ld, caption=""):
        self.ld = ld
        self.caption = caption
        self.freq = 0         # number of training instances this rule covers
        self.usability = 0.0  # percentage of training set covered by this rule

    def __str__(self):
        return self.caption

    def __call__(self, *args, **kwargs):
        return self.ld(*args, **kwargs)


