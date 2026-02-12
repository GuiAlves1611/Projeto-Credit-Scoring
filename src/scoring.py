import numpy as np


def fit_score_scale(p1, s1, p2, s2):
    """
    Ajusta A e B para:
        score = A - B * ln(odds),  odds = p/(1-p)
    Usando duas âncoras (p1->s1, p2->s2).
    """
    o1 = p1 / (1 - p1)
    o2 = p2 / (1 - p2)
    B = (s1 - s2) / (np.log(o2) - np.log(o1))
    A = s1 + B * np.log(o1)
    return A, B

def proba_to_score(p, A, B, clip_min=300, clip_max=850):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds = p / (1 - p)
    score = A - B * np.log(odds)
    return np.clip(score, clip_min, clip_max)

def rating(score, cuts):
    if score >= cuts["q90"]:
        return "A - Excelente"
    elif score >= cuts["q70"]:
        return "B - Bom"
    elif score >= cuts["q40"]:
        return "C - Regular"
    elif score >= cuts["q15"]:
        return "D - Risco"
    else:
        return "E - Alto Risco"


def decision_by_score(score, cuts):
    if score < cuts["cut_reprovado"]:
        return "Reprovado"
    elif score < cuts["cut_manual"]:
        return "Análise Manual"
    elif score < cuts["cut_restricao"]:
        return "Aprovado com Restrição"
    else:
        return "Aprovado"