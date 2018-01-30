# @Author:      HgS_1217_
# @Create Date: 2018/1/29

import xlrd
import numpy as np
import tensorflow as tf
from config import MAIN_PATH


def int_to_list(i, total):
    return [0] * (int(i) - 1) + [1] + [0] * (total - int(i))


def xlsx_read():
    data = xlrd.open_workbook("Problem_C_Data.xlsx")
    print("Read complete")
    table = data.sheets()[1]
    nrows, ncols = table.nrows, table.ncols
    rows = [table.row_values(i) for i in range(1, nrows)]
    v_max, dis_max = 20, 66000
    xs = np.array([int_to_list(row[2], 5) + int_to_list(row[3], 4) + int_to_list(row[4], 32) +
                   int_to_list(row[5], 32) + [row[1] / v_max, row[6] / dis_max] for row in rows])
    return xs


def main():
    xs = xlsx_read()

    ckpt = tf.train.get_checkpoint_state(MAIN_PATH)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    graph = tf.get_default_graph()

    x = graph.get_operation_by_name('input_x').outputs[0]
    y = tf.get_collection('pred_network')[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt.model_checkpoint_path)

        result = sess.run(y, feed_dict={x: xs, keep_prob: 1.0})
        print([r[0]*2000 for r in result])

if __name__ == '__main__':
    main()
