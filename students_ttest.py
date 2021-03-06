import numpy as np
from scipy import stats

# The t score is a ratio between the difference between two groups and the difference within the groups.
# The larger the t score, the more difference there is between groups.

# Low p-values are good; They indicate your data did not occur by chance.
# Example: a p-value of .01 means there is only a 1% probability that the results happened by chance.
# In most cases, a p-value of 0.05 (5%) is accepted to mean the data is valid.


def print_t_test(msg, a, b):

    print()
    print(msg)

    res_1 = stats.ttest_ind(a, b)
    res_2 = stats.ttest_ind(a, b, equal_var=False)

    print("Mean a:", np.mean(a))
    print("Mean b:", np.mean(b))
    print(res_1)
    print(res_2)
    print()


if __name__ == "__main__":

    # rvs1 = stats.norm.rvs(loc=5, scale=1, size=5)
    # rvs2 = stats.norm.rvs(loc=5, scale=1, size=5)
    # print(rvs1)
    # print(rvs2)

    # print_t_test("Example:", rvs1, rvs2)

    oh_32_g_0_99 = [39.33, 58.64, 47.75, 50.17, 52.67]
    oh_32_g_1 = [55.23, 46.13, 32.66, 61.54, 21.9]

    print_t_test("DQN Random - Gamma 0.99 vs 1:", oh_32_g_0_99, oh_32_g_1)

    ###

    oh_32_g_0_99_dddqn = [75.48, 75.46, 62.99, 86.45, 74.96]
    oh_32_g_1_dddqn = [76.23, 84.42, 73.6, 66.95, 82.0]

    print_t_test("DDDQN PER Random - Gamma 0.99 vs 1:", oh_32_g_0_99_dddqn, oh_32_g_1_dddqn)

    ###

    minmax_oh_g_0_99_dddqn = [82.52, 82.58, 83.52, 83.25, 79.54]
    minmax_oh_g_1_dddqn = [82.49, 79.5, 79.78, 84.59, 76.94]

    print_t_test("DDDQN PER Minmax OneHot - Gamma 0.99 vs 1:", minmax_oh_g_0_99_dddqn, minmax_oh_g_1_dddqn)

    ###

    minmax_2d_g_0_99_dddqn = [84.62, 83.75, 86.42, 83.96, 83.93]
    minmax_2d_g_1_dddqn = [85.48, 86.03, 85.1, 84.51, 85.21]

    print_t_test("DDDQN PER Minmax 2D - Gamma 0.99 vs 1:", minmax_2d_g_0_99_dddqn, minmax_2d_g_1_dddqn)

    ###
    raw_16 = [-24.14, -61.53, -45.97, -23.16, -81.05]
    oh_16 = [30.52, 22.81, 18.4, 7.73, 26.52]

    print_t_test("Raw vs OneHot dqn:", raw_16, oh_16)

    ###
    oh_32 = [55.23, 46.13, 32.66, 61.54, 21.9]

    print_t_test("16 vs 32 dqn:", oh_16, oh_32)

    # oh_16_dddqn = [25.45, 56.76, 42.72, 67.8, 45.41]

    # print_t_test("dqn vs dddqn 16:", oh_16, oh_16_dddqn)

    ###
    random_oh_32_dddqn = [76.23, 84.42, 73.6, 66.95, 82.0]

    print_t_test("dqn vs ddqn 32:", oh_32, random_oh_32_dddqn)

    # print_t_test("16 vs 32 dddqn:", oh_16_dddqn, random_oh_32_dddqn)

    ###

    minmax_oh_32 = [82.49, 79.5, 79.78, 84.59, 76.94]

    print_t_test("Random vs MinMax:", random_oh_32_dddqn, minmax_oh_32)

    ###
    minmax_2d = [85.48, 86.03, 85.1, 84.51, 85.21]

    print_t_test("OneHot vs 2D:", minmax_oh_32, minmax_2d)

    ###
    minmax_2d_0_2 = [98.42, 98.54, 98.44, 98.61, 98.41]

    print_t_test("MinMax 2D 0.0 vs 0.2:", minmax_2d, minmax_2d_0_2)

    ###
    self_0_0 = [97.2, 85.56, 92.1, 96.43, 93.33]
    self_0_2 = [98.06, 98.65, 98.64, 98.0, 98.48]

    print_t_test("Self 2D 0.0 vs 0.2:", self_0_0, self_0_2)
