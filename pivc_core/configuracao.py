"""
configuracao.py — Parâmetros fixos do sistema PIVC.

DISTANCIA_PADRAO_M: a captura das imagens é padronizada a 5 metros do
objeto. O valor fica FIXO para todas as imagens — por isso não aparece na
interface — e é registrado apenas nos metadados de cada experiência na
memória (memoria.json), pronto para cálculos de escala (área/volume de
entulho) em evoluções futuras do sistema.
"""

DISTANCIA_PADRAO_M = 5.0
