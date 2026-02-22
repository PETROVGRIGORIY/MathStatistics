import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
import math


def sample_from_exp_distribution(size: int):
    """для моделирования случайной величины разыгрываем равномерно
    распределённую сл.вел. ~ R(0,1);
    Выражаем x из значения функции распределения"""
    function_values_arr = np.random.uniform(0, 1, size)
    sample = -np.log(1 - function_values_arr)
    return sample

def _sample_mode(sample):
    """Мода выборки, если все элементы разные - вернёт None"""
    values, counts = np.unique(sample, return_counts=True)
    index_of_max_count = np.argmax(counts)

    mode_value = values[index_of_max_count]
    mode_count = counts[index_of_max_count]

    if mode_count == 1:
        return None
    return mode_value

def _sample_median(sample: np.ndarray):
    sorted_sample = np.sort(sample)
    if sorted_sample.size % 2 == 1:
        return sorted_sample[sorted_sample.size // 2]
    else:
        left_element = sorted_sample[sorted_sample.size // 2 - 1]
        right_element = sorted_sample[sorted_sample.size // 2]
        mean_val = (left_element + right_element) / 2
        return (mean_val)

def _mu_k(sample: np.ndarray, k: int):
    """к-й момент"""
    mean_val = np.mean(sample)
    return np.sum((sample - mean_val)**k) / sample.size

def _asymmetry_coefficient(sample: np.ndarray):
    return _mu_k(sample, 3) / (_mu_k(sample, 2))**(3/2)

def sample_information(sample: np.ndarray):
    """Возвращает моду, медиану, размах, оценку коэффициента асимметрии,
    если все элементы разные то в качестве моды вернётся None"""

    mode = _sample_mode(sample)
    if mode is None:
        print("Все элементы выборки разные, все являются модами")
    else:
        print(f"Мода: {mode}")
    
    median = _sample_median(sample)
    print(f"Медиана: {median}")

    sample_range = np.max(sample) - np.min(sample)
    print(f"Размах: {sample_range}")

    asm_coefficient = _asymmetry_coefficient(sample)
    print(f"Коэффициент асимметрии (оценка): {asm_coefficient}")

    return mode, median, sample_range, asm_coefficient

###################################

def empirical_distribution_function_plot(sample):
    abscissa, counts = np.unique(sample, return_counts=True)
    cum_counts = np.cumsum(counts) #кумулятивная сумма количества повторений
    ordinate = cum_counts / sample.size

    plt.figure(figsize=(10, 6))

    for i in range(len(abscissa)):
        if i == 0:
            #отступим на 20 процентов размаха со значением 0
            x_left = abscissa[0] - 0.2 * (abscissa[-1] - abscissa[0])
            plt.hlines(y=0, xmin=x_left, xmax=abscissa[0], 
                      colors='cornflowerblue', linewidth=2)
        else:
            plt.hlines(y=ordinate[i-1], xmin=abscissa[i-1], xmax=abscissa[i], 
                      colors='cornflowerblue', linewidth=2)

    #отступим на 20 процентов размаха со значением 1
    x_right = abscissa[-1] + 0.2 * (abscissa[-1] - abscissa[0])
    plt.hlines(y=1, xmin=abscissa[-1], xmax=x_right, 
              colors='cornflowerblue', linewidth=2)

    plt.scatter(
        abscissa, ordinate,
        s=50, facecolors='white', edgecolors='cornflowerblue',
        linewidth=2, zorder=5
    )
    
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title('Эмпирическая функция распределения')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.show()

def histogram(sample: np.ndarray):
    sorted_sample = np.sort(sample)
    k = int(1 + np.log2(sample.size)) #Количество интервалов

    plt.hist(
        sorted_sample,
        bins = k,
        density = True, #Нормировали частоту
        alpha = 0.8,
        color = 'cornflowerblue',
        edgecolor = 'blue',
        linewidth = 1.5
    )
    plt.xlabel('Значения выборки')
    plt.ylabel('Частота')
    plt.title('Гистограмма выборки')
    plt.show()

def boxplot(sample):
    plt.figure(figsize=(8, 6))
    plt.boxplot(
        sample,
        vert = False,
        patch_artist=True,
        boxprops=dict(facecolor="cornflowerblue"),
        medianprops=dict(color="black"),
    )
    plt.title('boxplot выборки')
    plt.show()

#############################

def _mean_value_bootstrap(sample: np.ndarray, n: int) -> np.ndarray:
    """аргумент n - количество подвыборок"""
    mean_values = np.zeros(n)

    for i in range(n):
        bootstrap_sample = np.random.choice(
            sample,
            size = sample.size,
            replace=True  #c повторениями элементов
        )
        mean = np.mean(bootstrap_sample)
        mean_values[i] = mean

    return mean_values

def compare_mean_cpt_bootstrap(sample: np.ndarray, n: int):
    """n - количество подвыборок для bootstrap"""
    mu = 1
    sigma = 0.2  # 1/5

    abscissa = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 1000)
    ordinate = scp.stats.norm.pdf(abscissa, loc=mu, scale=sigma)

    bootstrap_mean_values = _mean_value_bootstrap(sample, n)
    k = int(1 + np.log2(bootstrap_mean_values.size)) #Количество интервалов

    plt.hist(
        bootstrap_mean_values,
        bins = k,
        density = True, #Нормировали частоту
        alpha = 0.8,
        color = 'cornflowerblue',
        edgecolor = 'blue',
        linewidth = 1.5,
        label="bootstrap"
    )
    plt.plot(
        abscissa, 
        ordinate, 
        linewidth=2.5, 
        label='ЦПТ',
        color='black'
    )

    plt.title('Оценка среднего арифметического по ЦПТ и bootstrap')
    plt.show()

##############################################

def _bootstrap_asm_coefficient(sample: np.ndarray, n: int):
    """n - количество подвыборок для bootstrap"""
    asm_coefficient_values = np.zeros(n)

    for i in range(n):
        bootstrap_sample = np.random.choice(
            sample,
            size = sample.size,
            replace=True  #c повторениями элементов
        )
        asm_coefficient = _asymmetry_coefficient(bootstrap_sample)
        asm_coefficient_values[i] = asm_coefficient
    
    return asm_coefficient_values

def bootstrap_asm_coefficient_hist(sample: np.ndarray, n: int):
    """n - количество подвыборок для bootstrap"""
    asm_coefficient_values = _bootstrap_asm_coefficient(sample, n)
    k = int(1 + np.log2(asm_coefficient_values.size)) #Количество интервалов

    #сразу посчитаем вероятность того, что коэффициент асимметрии будет меньше 1
    #Количество подходящих делим на общее количество
    p = (np.sum(asm_coefficient_values < 1)) / asm_coefficient_values.size
    print(f"Вероятность того, что коэффициент асимметрии < 1: {p}")

    plt.hist(
        asm_coefficient_values ,
        bins = k,
        density = True, #Нормировали частоту
        alpha = 0.8,
        color = 'cornflowerblue',
        edgecolor = 'blue',
        linewidth = 1.5,
        label="bootstrap"
    )

    plt.title('Оценка коэффициента асимметрии по bootstrap')
    plt.show()

########################################

def _bootstrap_median(sample: np.ndarray, n: int):
    """n - количество подвыборок для bootstrap"""
    median_values = np.zeros(n)
    
    for i in range(n):
        bootstrap_sample = np.random.choice(
            sample,
            size = sample.size,
            replace=True  #c повторениями элементов
        )
        median = _sample_median(bootstrap_sample)
        median_values[i] = median

    return median_values

def p(abscissa: np.ndarray):
    array = np.copy(abscissa)
    array[array >= 0] = np.exp(-array[array >= 0])
    array[array < 0] = 0
    return array

def F(abscissa: np.ndarray):
    return 1 - np.exp(-abscissa) 

def density_function_k_order_statistic(
        n: int, k: int, abscissa: np.ndarray
    ):

    factorial_part = math.factorial(n)/(math.factorial(k-1)*math.factorial(n-k))
    function_part = (F(abscissa)**(k-1)) * ((1 - F(abscissa)) ** (n-k)) * p(abscissa)

    return factorial_part * function_part

def compare_bootstrap_median(sample: np.ndarray, n: int):
    bootstrap_median_values = _bootstrap_median(sample, n)
    k = int(1 + np.log2(bootstrap_median_values.size)) #Количество интервалов

    plt.hist(
        bootstrap_median_values,
        bins = k,
        density = True, #Нормировали частоту
        alpha = 0.8,
        color = 'cornflowerblue',
        edgecolor = 'blue',
        linewidth = 1.5,
    )

    abscissa = np.linspace(
        np.min(bootstrap_median_values),
        np.max(bootstrap_median_values),
        1000)
    ordinate = density_function_k_order_statistic(25, 13, abscissa)

    plt.plot(
        abscissa, 
        ordinate, 
        linewidth=2.5, 
        color='black'
    )
    plt.show()
