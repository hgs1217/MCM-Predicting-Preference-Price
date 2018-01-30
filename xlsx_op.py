# @Author:      HgS_1217_
# @Create Date: 2018/1/29

import xlrd
import numpy as np
from fcnet import FCNet


def int_to_list(i, total):
    return [0] * (int(i) - 1) + [1] + [0] * (total - int(i))


def xlsx_read():
    data = xlrd.open_workbook("Problem_C_Data.xlsx")
    # data = xlrd.open_workbook("test.xlsx")
    print("Read complete")
    table = data.sheets()[0]
    nrows, ncols = table.nrows, table.ncols
    rows = [table.row_values(i) for i in range(1, nrows)]
    v_max, dis_max, price_max = 20, 66000, 2000
    xs = np.array([int_to_list(row[2], 5) + int_to_list(row[3], 4) + int_to_list(row[4], 32) +
                   int_to_list(row[5], 32) + [row[1] / v_max, row[6] / dis_max] for row in rows])
    ys = np.array([[row[7] / price_max] for row in rows])
    return xs, ys


def main():
    xs, ys = xlsx_read()
    raws, labels = xs[:-10000], ys[:-10000]
    test_raws, test_labels = xs[-10000:], ys[-10000:]
    fcnet = FCNet(raws, labels, test_raws, test_labels, 0.8, 5000, 20000, start_step=20000)
    fcnet.train()


if __name__ == '__main__':
    main()
