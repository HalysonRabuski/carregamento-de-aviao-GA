from random import Random
from time import time
from math import cos
from math import pi
from inspyred import ec
from inspyred.ec import terminators
import numpy as np
import os


def generate_(random, args):
    size = args.get('num_inputs', 12)
    return [random.randint(0, 16000) for i in range(size)]


def evaluate_(candidates, args):
    fitness = []
    for cs in candidates:
        fit = perform_fitness(cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6], cs[7], cs[8], cs[9], cs[10], cs[11])
        fitness.append(fit)
    return fitness

def perform_fitness(C1D,C1C,C1T,C2D,C2C,C2T,C3D,C3C,C3T,C4D,C4C,C4T):
    C1D = np.round(C1D)
    C1C = np.round(C1C)
    C1T = np.round(C1T)

    C2D = np.round(C2D)
    C2C = np.round(C2C)
    C2T = np.round(C2T)

    C3D = np.round(C3D)
    C3C = np.round(C3C)
    C3T = np.round(C3T)

    C4D = np.round(C4D)
    C4C = np.round(C4C)
    C4T = np.round(C4T)

    fit = float((0.31*C1D + 0.31*C1C + 0.31*C1T + 0.38*C2D +  0.38*C2C + 0.38*C2T + 0.35*C3D + 0.35*C3C + 0.35*C3T + 0.285*C4D + 0.285*C4C + 0.285*C4T) / 12151.56)
    
    ## volume dianteiro
    h1 = np.maximum(0, float((0.48*C1D + 0.65*C2D + 0.58*C3D + 0.39*C4D) - 6800)) / 680 
    ## peso dianteiro
    h2 = np.maximum(0, float((C1D + C2D + C3D + C4D) - 10000)) / 1000
## -------------------------------------------------------------------------------------- ##
    ## volume central
    h3 = np.maximum(0, float((0.48*C1C + 0.65*C2C + 0.58*C3C + 0.39*C4C) - 8700)) / 870
    ## peso central
    h4 = np.maximum(0, float((C1C + C2C + C3C + C4C) - 16000)) / 1600
## -------------------------------------------------------------------------------------- ##
    ## volume traseiro
    h5 = np.maximum(0, float((0.48*C1T + 0.65*C2T + 0.58*C3T + 0.39*C4T) - 5300)) / 530 
    ## peso traseiro
    h6 = np.maximum(0, float((C1T + C2T + C3T + C4T) - 8000)) / 800

    #limite disponível de cada carga
    h7 = np.maximum(0, float((C1D + C1C + C1T) - 18000)) / 1800
    h8 = np.maximum(0, float((C2D + C2C + C2T) - 15000)) / 1500
    h9 = np.maximum(0, float((C3D + C3C + C3T) - 23000)) / 2300
    h10 = np.maximum(0, float((C4D + C4C + C4T) - 12000)) / 1200


    fit = fit - (h1 + h2 + h3 + h4 + h5 + h6 + h7 + h8 + h9 + h10) 

    return fit


def solution_evaluation(C1D,C1C,C1T,C2D,C2C,C2T,C3D,C3C,C3T,C4D,C4C,C4T):

    C1D = np.round(C1D)
    C1C = np.round(C1C)
    C1T = np.round(C1T)

    C2D = np.round(C2D)
    C2C = np.round(C2C)
    C2T = np.round(C2T)

    C3D = np.round(C3D)
    C3C = np.round(C3C)
    C3T = np.round(C3T)

    C4D = np.round(C4D)
    C4C = np.round(C4C)
    C4T = np.round(C4T)

    print("..:RESUMO DO CARREGAMENTO:..")
    print("Lucro Total", float(0.31*C1D + 0.31*C1C + 0.31*C1T + 0.38*C2D +  0.38*C2C + 0.38*C2T + 0.35*C3D + 0.35*C3C + 0.35*C3T + 0.285*C4D + 0.285*C4C + 0.285*C4T))
    print("Volume dianteiro ocupado", float(0.48*C1D + 0.65*C2D + 0.58*C3D + 0.39*C4D))
    print("Volume central ocupado", float(0.48*C1C + 0.65*C2C + 0.58*C3C + 0.39*C4C))
    print("Volume traseiro ocupado", float(0.48*C1T + 0.65*C2T + 0.58*C3T + 0.39*C4T))
    print("Volume total ocupado", float(0.48*C1D + 0.65*C2D + 0.58*C3D + 0.39*C4D + 0.48*C1C + 0.65*C2C + 0.58*C3C + 0.39*C4C + 0.48*C1T + 0.65*C2T + 0.58*C3T + 0.39*C4T))
    print("Peso total da dianteira", float(C1D + C2D +C3D + C4D))
    print("Peso total da central", float(C1C + C2C +C3C + C4C))
    print("Peso total da traseira", float(C1T + C2T +C3T + C4T))
    
    print("C1 DIANTEIRO",C1D)
    print("C1 CENTRAL",C1C)
    print("C1 TRASEIRO",C1T)
    print("CARGA TOTAL C1: ",C1D+C1C+C1T)

    print("----------------------")

    print("C2 DIANTEIRO",C2D)
    print("C2 CENTRAL",C2C)
    print("C2 TRASEIRO",C2T)
    print("CARGA TOTAL C2: ",C2D+C2C+C2T)
    print("----------------------")

    print("C3 DIANTEIRO",C3D)
    print("C3 CENTRAL",C3C)
    print("C3 TRASEIRO",C3T)
    print("CARGA TOTAL C3: ",C3D+C3C+C3T)
    print("----------------------")

    print("C4 DIANTEIRO",C4D)
    print("C4 CENTRAL",C4C)
    print("C4 TRASEIRO",C4T)
    print("CARGA TOTAL C4: ",C4D+C4C+C4T)

    # print("Tempo de utilização semanal maquina (MAX 60h): ", float((6*L + 5*S) / 100))
    # print("Espaço utilizado do deposito (MAX. 1500)", float(10*L + 20*S))
    # print("Produção de Garrafas tipo LEITE produzidas (MAX. 800): ", L)
    # print("Produção de garrafas do tipo SUCO produzidas (MAX .750): ", S)


def main():
    rand = Random()
    rand.seed(int(time()))

    ea = ec.GA(rand)
    ea.selector = ec.selectors.tournament_selection
    ea.variator = [ec.variators.uniform_crossover,
                   ec.variators.gaussian_mutation]
    ea.replacer = ec.replacers.steady_state_replacement

    ea.terminator = terminators.generation_termination

    ea.observer = [ec.observers.stats_observer, ec.observers.file_observer]

    final_pop = ea.evolve(generator=generate_,
                          evaluator=evaluate_,
                          pop_size=10000,
                          maximize=True,
                          bounder=ec.Bounder(0, 16000),
                          max_generations=10000,
                          num_imputs=12,
                          crossover_rae=1.0,
                          num_crossover_points=1,
                          mutation_rate=0.05,
                          num_elites=1,
                          num_selected=12,
                          tournament_size=12,
                          statistics_file=open("aviao.csv", "w"),
                          individuals_file=open("aviao_dados.csv", "w"))

    final_pop.sort(reverse=True)
    print(final_pop[0])

    perform_fitness(final_pop[0].candidate[0], final_pop[0].candidate[1],final_pop[0].candidate[2],final_pop[0].candidate[3],final_pop[0].candidate[4],final_pop[0].candidate[5],final_pop[0].candidate[6],final_pop[0].candidate[7],final_pop[0].candidate[8],final_pop[0].candidate[9],final_pop[0].candidate[10],final_pop[0].candidate[11])
    solution_evaluation(final_pop[0].candidate[0], final_pop[0].candidate[1], final_pop[0].candidate[2], final_pop[0].candidate[3], final_pop[0].candidate[4], final_pop[0].candidate[5], final_pop[0].candidate[6], final_pop[0].candidate[7], final_pop[0].candidate[8], final_pop[0].candidate[9], final_pop[0].candidate[10], final_pop[0].candidate[11])


main()
