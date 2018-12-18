def test(jobs=100, **kwargs):
    print('jobs:', jobs)


def upper_test(**kwargs):
    test(**kwargs)
    print(kwargs)


upper_test(jobs=100, important=100)
