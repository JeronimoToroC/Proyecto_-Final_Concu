from src.controllers.manager import Manager

from src.controllers.strategies.q_nodes import QNodes


def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    estado_inicial = "100000000000000"
    condiciones =    "111011111111111"
    alcance =        "111111111111111"
    mecanismo =      "111111111111111"

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_qn = QNodes(gestor_sistema)

    sia_uno = analizador_qn.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )

    print(sia_uno)
