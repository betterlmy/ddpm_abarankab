import matplotlib.pyplot as plt


def extract(a, t, x_shape):
    b, *_ = t.shape  # *_忽略其他元素
    out = a.gather(-1, t)
    # gather从输入 a 的最后一个维度（-1），根据指定的索引 t 抽取元素。
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def plot(x, y, x_name="x", y_name="y"):
    plt.plot(x, y)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()

def printArgs(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))


if __name__ == "__main__":
    plot([1, 2], [2, 2])
