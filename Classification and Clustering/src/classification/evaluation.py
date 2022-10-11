import typing as th


def accuracy(y, y_hat) -> float:
    # todo: for you to implement
    count = 0
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            count += 1
    ac = count/len(y)
    return ac


def f1(y, y_hat, alpha: float = 0.5, beta: float = 1.):
    # todo: for you to implement
    p0, p1 = precision(y, y_hat)
    r0, r1 = precision(y, y_hat)
    fm0 = (beta*beta + 1)*p0*r0/(beta*beta*p0+r0)
    fm1 = (beta*beta + 1)*p1*r1/(beta*beta*p1+r1)
    return fm0, fm1


def precision(y, y_hat) -> float:
    # todo: for you to implement
    tp0 = 0
    tp1 = 0
    fp0 = 0
    fp1 = 0
    for i in range(len(y_hat)):
        if(y_hat[i] == 0):
            if(y[i] == 0):
                tp0 += 1
            else:
                fp0 += 1
        else:
            if(y[i] == 1):
                tp1 += 1
            else:
                fp1 += 1
    p0 = tp0/(tp0+fp0)
    p1 = tp1/(tp1+fp1)
    return p0, p1


def recall(y, y_hat) -> float:
    # todo: for you to implement
    tp0 = 0
    tp1 = 0
    fn0 = 0
    fn1 = 0
    for i in range(len(y)):
        if y[i] == 0:
            if y_hat[i] == 0:
                tp0 += 1
            else:
                fn0 += 1
        else:
            if y_hat[i] == 1:
                tp1 += 1
            else:
                fn1 += 1
    r0 = tp0/(tp0+fn0)
    r1 = tp1/(tp1+fn1)
    return r0, r1

evaluation_functions = dict(accuracy=accuracy, f1=f1, precision=precision, recall=recall)


def evaluate(y, y_hat) -> th.Dict[str, float]:
    """
    :param y: ground truth
    :param y_hat: model predictions
    :return: a dictionary containing evaluated scores for provided values
    """
    acc = accuracy(y, y_hat)
    fm0, fm1 = f1(y, y_hat)
    p0, p1 = precision(y, y_hat)
    r0, r1 = recall(y, y_hat)
    evaluation_functions = {
        'ACC' : acc,
        'F1_pos' : fm1,
        'prec_pos' : p1,
        'reca_pos' : r1,
        'F1_neg' : fm0,
        'prec_neg' : p0,
        'reca_neg' : r0
        }
    return evaluation_functions
