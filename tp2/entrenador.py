from sklearn import tree
from joblib import dump, load

# dataset
"""
EJEMPLO PROFESOR
X = [
    [6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37],
    [6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04],
    [6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37],
    [1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16, 9.67218398e-18],
    [8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37],
    [6.53608067e-04, 6.07480284e-16, 9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37],
    [9.67218398e-18, 1.40311655e-19, -1.18450102e-37, 8.60883492e-28, -1.12639633e-37, 6.53608067e-04, 6.07480284e-16],
]

# etiquetas, correspondientes a las muestras
Y = [1, 1, 1, 2, 2, 3, 3]
"""

def main():
    with open("descriptores.txt", "r") as f:
        lineas = f.readlines()

    X = eval(lineas[0].strip())
    Y = eval(lineas[1].strip())

    # entrenamiento
    clasificador = tree.DecisionTreeClassifier(criterion="entropy",
                                               max_depth=2,
                                               min_samples_split=4,
                                               min_samples_leaf=2,
                                               ).fit(X, Y)

    # visualización del árbol de decisión resultante
    tree.plot_tree(clasificador)

    # guarda el modelo en un archivo
    dump(clasificador, 'filename.joblib')

if __name__ == "__main__":
    main()
