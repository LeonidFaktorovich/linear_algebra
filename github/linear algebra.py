# импортируем модуль чтобы было удобно работать с дробями, используется только в строчке 182
from fractions import Fraction


# Функция для нахождения всех перестановок на n элементах, 0 < n < 100
def all_perm(n):
    if n == 1:
        return [[1]]
    result = []
    last_res = all_perm(n - 1)
    # перебираем все перестановки на n - 1 элементе и вставляем на всевозможные позиции число n
    for permutation in last_res:
        for pos_n in range(n):
            result.append(permutation[: pos_n] + [n] + permutation[pos_n:])
    return result


# Функция, которая работает как сортировка слиянием, возвращая отсортированную последовательность и
# количество инверсий
def num_of_inversions(perm):
    if len(perm) == 1:
        return [perm, 0]

    result = 0
    medium = len(perm) // 2
    # делим последовательность пополам, затем сортируем их и находим число инверсий в каждой
    first_result = num_of_inversions(perm[:medium])
    second_result = num_of_inversions(perm[medium:])
    result += first_result[1] + second_result[1]
    index_first = 0
    index_second = 0
    new_sequence = []
    while index_first != len(first_result[0]) or index_second != len(second_result[0]):
        if index_first == len(first_result[0]):
            new_sequence.append(second_result[0][index_second])
            index_second += 1
        elif index_second == len(second_result[0]):
            new_sequence.append(first_result[0][index_first])
            # раз до этого элемента мы поставили index_second элементов из второй части,
            # то с каждым из них он стоит в порядке убывания, поэтому ко множеству инверсий добавляется
            # index_second пар с участием числа first_result[0][index_fir   st]
            result += index_second
            index_first += 1
        else:
            if first_result[0][index_first] < second_result[0][index_second]:
                new_sequence.append(first_result[0][index_first])
                # раз до этого элемента мы поставили index_second элементов из второй части,
                # то с каждым из них он стоит в порядке убывания
                result += index_second
                index_first += 1
            else:
                new_sequence.append(second_result[0][index_second])
                index_second += 1
    return [new_sequence, result]


# благодаря количеству инверсий, находим знак перестановки
def sign(perm):
    # если инверсий четно, то знак 1
    if num_of_inversions(perm)[1] % 2 == 0:
        return 1
    # иначе знак -1
    else:
        return -1


# функция находит произведени перестановок одинакового размера perm_2 * perm_1, у которых верхняя строчка это 1,2,...,n
# а нижняя perm_1 и perm_2
# вывод можно воспринимать, как нижнюю строчку в произведении перестановок, где верзняя это 1,2,3,...,n
def multiplication_perm(perm_1, perm_2):
    n = len(perm_1)
    result = [None] * n
    for i in range(n):
        # заметим, что число равное i сначала переходит в perm_1[i - 1] а уже это число в perm_2[perm_1[i - 1] - 1]
        # так как мы берем не i, а i - 1, то получаем следующую формулу
        result[i] = perm_2[perm_1[i] - 1]
    return result


# Считаем определитель квадратной матрицы в котором среди элементов могут быть x, важно что это должен быть именно х,
# а не -x или ax, где а - произвольный коэффициент
# возвращает эта функция коэффициенты при x^0 , x^1 , x^2 ,... соответсвенно
def det(matrix):
    n = len(matrix)
    # так как определитель этой матрицы - многочлен от х не более чем n степени, то будем в result хранить коэффициенты
    # этого многочлена, где result[0] - свободный коэффициент, result[1] - коэффициент при x и так далее
    result = [0] * (n + 1)
    all_permutations = all_perm(n)
    # для каждой перестановки ищем, какая при подсчете получается степень x, затем прибавляем к соответствующему коэф.
    for permutation in all_permutations:
        # находим все "элементы" текущей перестановки в матрице, эти элементы - либо числа, либо 'x'
        # так как индексация столбцов начинается с 0, то мы должны взять столбик permutation[i] - 1
        our_numbers = [matrix[i][permutation[i] - 1] for i in range(n)]
        # находим количество 'x'
        count_of_x = our_numbers.count('x')
        current_value = 1
        # находим произведение чисел
        for num in our_numbers:
            if num != 'x':
                current_value *= num
        # умножаем на знак
        result[count_of_x] += current_value * sign(permutation)
    return result


# функция получается на вход 2 матрицы, которые гарантированно можно перемножить и возвращает matrix_1 * matrix_2
def multiplication_matrix(matrix_1, matrix_2):
    # размеры matrix_1 - n на k
    # размеры matrix_2 - k на m
    n = len(matrix_1)
    k = len(matrix_1[0])
    m = len(matrix_2[0])
    result = [[0] * m for _ in range(n)]
    for row in range(n):
        for col in range(m):
            for i in range(k):
                result[row][col] += matrix_1[row][i] * matrix_2[i][col]
    return result


# функция получается на вход 2 матрицы, которые гарантированно можно сложить и возвращает matrix_1 + matrix_2
def sum_matrix(matrix_1, matrix_2):
    n = len(matrix_1)
    m = len(matrix_1[0])
    result = [[None] * m for _ in range(n)]
    for row in range(n):
        for col in range(m):
            result[row][col] = matrix_1[row][col] + matrix_2[row][col]
    return result


# функция получает на вход матрицу, а также 2 списка, список строк и список столбцов, которые нужно удалить, индексация
# строк и столбцов начинается с 0, гарантируется что эти строки и столбцы можно удалить, в каждом из списков нет
# повторений
def del_rows_and_cols(matrix, rows, cols):
    # копируем матрицу
    new_matrix = [matrix[i][::] for i in range(len(matrix))]
    # сортируем индексы строк и столбцов в порядке убывания, чтобы после удаления индексы оставшихся строк и столбцов
    # не менялись
    rows.sort(reverse=True)
    cols.sort(reverse=True)
    # удаляем строки
    for row in rows:
        del new_matrix[row]
    # удаляем столбцы
    for col in cols:
        for i in range(len(new_matrix)):
            del new_matrix[i][col]
    return new_matrix


# функция получает на вход матрицу и возвращает присоединенную матрицу для полученной
def find_adjoint_matrix(matrix):
    n = len(matrix)
    m = len(matrix[0])
    adjoint_matrix = [[None] * m for _ in range(n)]
    for row in range(n):
        for col in range(m):
            # удаляем строчку равную столбцу элемента и столбик равный строчке элемента
            matrix_after_del = del_rows_and_cols(matrix, [col], [row])
            # находим определитель нужной матрицы и домножаем на коэффициент
            adjoint_matrix[row][col] = ((-1) ** (row + col)) * det(matrix_after_del)[0]
    return adjoint_matrix


# функция получает на вход матрицу и возвращает обратную матрицу, если такая есть, иначе выводит det = 0
def find_inverse_matrix(matrix):
    # находим определитель матрицы matrix
    det_matrix = det(matrix)[0]
    if det_matrix == 0:
        print('det = 0')
        return
    n = len(matrix)
    m = len(matrix[0])
    inverse_matrix = [[None] * m for _ in range(n)]
    adjoint_matrix = find_adjoint_matrix(matrix)
    # делим каждый элемент присоединенной матрицы на определитель матрицы
    for row in range(n):
        for col in range(m):
            # проверяем, если у нас целые числа, то мы можем работать с дробями
            if int(adjoint_matrix[row][col]) == adjoint_matrix[row][col] and int(det_matrix) == det_matrix:
                inverse_matrix[row][col] = Fraction(int(adjoint_matrix[row][col]), int(det_matrix))
            else:
                inverse_matrix[row][col] = adjoint_matrix[row][col] / det_matrix
    return inverse_matrix


# функция получает 2 параметра n и k и возвращает всевозможные выборы k чисел из множества от 0 до n - 1 без учета
# порядка, гарантируется что 0 <= k <= n
def variations(n, k):
    # рассматриваем крайние случаи, чтобы сделать рекурсию
    if k == 0:
        return [[]]
    if k == 1:
        return [[i] for i in range(n)]
    if n == k:
        return [[i for i in range(n)]]
    result = []
    # добавляем все варианты, где мы выбрали число n - 1
    result += variations(n - 1, k - 1)
    for var in result:
        var.append(n - 1)
    # добавляем все варианты, где мы не выбрали число n - 1
    result += variations(n - 1, k)
    return result


# функция получается на вход квадратную матрицу и возвращает коэффициенты характеристического многочлена этой матрицы
# соответственно перед lambda^0, lambda^1 ,...
def char_pol(matrix):
    n = len(matrix)
    # задаем в какой степени в коэффициент будет входить -1
    result = [(-1) ** i for i in range(n, -1, -1)]
    # для каждой степени lambda кроме n посчитаем коэффициент, перед lambda^n коэффициент точно равен 1
    for power_lambda in range(0, n):
        # переменная, в которой будет сумма определителей матриц n - power_lambda на n - power_lambda
        current_factor = 0
        # находим всевозможные варианты какие столбцы и строки можно удалить,
        # чтобы получить lambda в степени power_lambda
        all_var = variations(n, power_lambda)
        # для каждого такого варианты считает определитель матрицы после удаления строк и столбцов
        # и прибавляем его к current_factor
        for variation in all_var:
            # находим матрицу после удаления строк и столбцов
            matrix_after_del = del_rows_and_cols(matrix, variation, variation)
            # прибавляем ее определитель
            current_factor += det(matrix_after_del)[0]
        # находим конечный коэффициент в характеристическом многочлене
        result[power_lambda] *= current_factor
    return result
