import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster

CRITERION_TYPE = {
    'c1': 'certain', 'c2': 'certain', 'c3': 'certain', 'c5': 'certain',
    'c4': 'epistemic', 'c6': 'epistemic', 'c7': 'epistemic',
    'c8': 'aleatoric', 'c9': 'aleatoric',
}
COLOR_MAP = {'certain': '#888888', 'epistemic': '#4472C4', 'aleatoric': '#E06C00'}
DASH_MAP  = {'certain': 'solid',   'epistemic': 'solid',   'aleatoric': 'dash'}
# Remplissages clairs associés (non fournis par l'énoncé, dérivés pour les fonds/fills)
FILL_MAP  = {'certain': '#e6e6e6', 'epistemic': '#DEEAF1', 'aleatoric': '#FCE4D6'}
# DASH_MAP utilise le vocabulaire Plotly ("solid"/"dash") ; le CSS border-style
# attend "solid"/"dashed" — petite table de conversion entre les deux, utilisée
# partout où l'incertitude est rendue en HTML/CSS (légende, matrice).
CSS_BORDER_STYLE = {"solid": "solid", "dash": "dashed"}

def crit_type(c):
    """Type d'incertitude d'un critère : 'certain' / 'epistemic' / 'aleatoric'."""
    return CRITERION_TYPE.get(c, 'certain')

def crit_color(c):
    return COLOR_MAP[crit_type(c)]

def crit_dash(c):
    return DASH_MAP[crit_type(c)]

def crit_fill(c):
    return FILL_MAP[crit_type(c)]

# ── Chemin vers le fichier Excel ──────────────────────────────────────────────
# os.path.abspath → donne le chemin complet du fichier app.py
# os.path.dirname → prend juste le dossier (sans le nom du fichier)
# os.path.join    → colle le dossier + le nom du fichier Excel
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "3D_Printer_MCDA_Dataset_Tomelleri2025.xlsx")

# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

# Lire la feuille "Numeric Matrix" du fichier Excel (9 lignes = 9 alternatives)
num = pd.read_excel(DATA_PATH, sheet_name="Numeric Matrix", nrows=9)
# Renommer les colonnes pour qu'elles soient plus claires
num.columns = ["Alt","c1","c2","c3","c4_min","c4_max","c5","c6_min","c6_max","c7_min","c7_max"]
# Convertir toutes les colonnes (sauf Alt) en nombres
for c in num.columns[1:]:
    num[c] = pd.to_numeric(num[c], errors="coerce")

# Liste des 7 critères (utilisés dans le calcul TOPSIS)
CRITERIA = ["c1","c2","c3","c4","c5","c6","c7"]
BENEFIT  = ["c1","c4","c5"]        # critères à maximiser (plus grand = mieux)
COST     = ["c2","c3","c6","c7"]   # critères à minimiser (plus petit = mieux)
# UNCERTAIN/FIXED/ALEATORY dérivés de CRITERION_TYPE (source unique de vérité)
UNCERTAIN = [c for c, t in CRITERION_TYPE.items() if t == "epistemic"]  # ["c4","c6","c7"]
FIXED     = [c for c, t in CRITERION_TYPE.items() if t == "certain"]    # ["c1","c2","c3","c5"]

# Critères supplémentaires — AFFICHAGE SEULEMENT, PAS dans TOPSIS
EXTRA_CRITERIA = ["c8","c9"]        # c8 Rugosité Ra, c9 Porosité
ALEATORY = [c for c, t in CRITERION_TYPE.items() if t == "aleatoric"]   # ["c8","c9"]
EXTRA_BENEFIT = []                  # aucun critère supplémentaire à maximiser
EXTRA_COST    = ["c8","c9"]         # c8 = rugosité Ra → Min↘ ; c9 = porosité → Min↘
FULL_CRITERIA = CRITERIA + EXTRA_CRITERIA   # 9 critères pour l'affichage (scatter, parcoords)

# Liste des 9 alternatives (a1 à a9)
ALTS = num["Alt"].tolist()
N_ALT = len(ALTS)    # = 9
N_CRIT = len(CRITERIA)  # = 7

# Noms complets des critères (pour les titres des axes)
LABELS = {
    "c1":"c1 Print Volume (L)",  "c2":"c2 Feature Res. (µm)",
    "c3":"c3 Layer Thick. (µm)", "c4":"c4 VED (J/mm³)",
    "c5":"c5 Heated Bed",        "c6":"c6 Acq. Cost (€×1000)",
    "c7":"c7 Op. Cost (€/h)",
    "c8":"c8 Rugosité Ra (µm)",  "c9":"c9 Porosité (%)",
}
# Noms courts des critères (pour les graphes)
SHORT = {
    "c1":"Volume","c2":"Résolution","c3":"Épaisseur","c4":"VED",
    "c5":"Lit chauff","c6":"Coût acq","c7":"Coût op",
    "c8":"Rugosité","c9":"Porosité",
}

def crit_marker(c):
    return ""

def crit_label(c):
    return f"{c} — {SHORT[c]}"

# Couleurs pour chaque alternative (palette pastel — 9 teintes distinctes)
ALT_COLOR = {
    "a1":"#7FB3D5","a2":"#F5B041","a3":"#EC7063","a4":"#48C9B0",
    "a5":"#82C091","a6":"#F7DC6F","a7":"#BB8FCE","a8":"#F1948A","a9":"#B9967A",
}

# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION + TOPSIS
# ═══════════════════════════════════════════════════════════════════════════════

N_SIM = 100_000  # nombre de simulations Monte Carlo
rng = np.random.default_rng(42)  # générateur de nombres aléatoires (42 = graine fixe pour reproductibilité)

# Matrices min et max pour chaque alternative et chaque critère (9 × 7)
# Pour critères fixes : min = max = valeur nominale (pas d'incertitude)
mat_min = np.zeros((N_ALT, N_CRIT))
mat_max = np.zeros((N_ALT, N_CRIT))

# Remplir les matrices min/max depuis le fichier Excel
for i, row in num.iterrows():
    mat_min[i,0] = mat_max[i,0] = row["c1"]   # c1 fixe
    mat_min[i,1] = mat_max[i,1] = row["c2"]   # c2 fixe
    mat_min[i,2] = mat_max[i,2] = row["c3"]   # c3 fixe
    mat_min[i,3] = row["c4_min"];  mat_max[i,3] = row["c4_max"]  # c4 incertain
    mat_min[i,4] = mat_max[i,4] = row["c5"]   # c5 fixe
    mat_min[i,5] = row["c6_min"];  mat_max[i,5] = row["c6_max"]  # c6 incertain
    mat_min[i,6] = row["c7_min"];  mat_max[i,6] = row["c7_max"]  # c7 incertain

# Valeurs nominales = milieu de l'intervalle [min, max]
mat_nom = (mat_min + mat_max) / 2.0

# Poids AHP issus de l'article Tomelleri 2025 (somme = 1)
weights_ahp = np.array([0.1027, 0.3973, 0.1348, 0.2321, 0.0251, 0.0778, 0.0302])

def topsis_score(matrix, w):
    """TOPSIS avec normalisation vectorielle. Retourne scores de closeness (entre 0 et 1)."""
    # Étape 1 : Normalisation vectorielle (diviser chaque colonne par sa norme)
    denom = np.sqrt((matrix**2).sum(axis=0))
    denom[denom == 0] = 1  # éviter division par zéro
    r = matrix / denom
    # Étape 2 : Pondération (multiplier par les poids)
    v = r * w
    # Étape 3 : Trouver la solution idéale (meilleure) et anti-idéale (pire)
    ideal = np.zeros(N_CRIT)
    anti  = np.zeros(N_CRIT)
    for j, c in enumerate(CRITERIA):
        if c in BENEFIT:
            ideal[j] = v[:,j].max()  # bénéfice → idéal = max
            anti[j]  = v[:,j].min()  # bénéfice → anti-idéal = min
        else:
            ideal[j] = v[:,j].min()  # coût → idéal = min
            anti[j]  = v[:,j].max()  # coût → anti-idéal = max
    # Étape 4 : Calculer les distances à l'idéal et à l'anti-idéal
    d_plus  = np.sqrt(((v - ideal)**2).sum(axis=1))  # distance à l'idéal
    d_minus = np.sqrt(((v - anti)**2).sum(axis=1))   # distance à l'anti-idéal
    # Étape 5 : Score de closeness (plus proche de 1 = meilleur)
    denom_s = d_plus + d_minus
    denom_s[denom_s == 0] = 1
    return d_minus / denom_s

# Tableaux pour stocker les résultats des 100 000 simulations
mc_scores   = np.zeros((N_SIM, N_ALT))              # scores TOPSIS de chaque simulation
mc_ranks    = np.zeros((N_SIM, N_ALT), dtype=int)   # rangs de chaque simulation
mc_matrices = np.zeros((N_SIM, N_ALT, N_CRIT))      # matrices brutes pour calculs statistiques

# Boucle principale : lancer les N_SIM simulations Monte Carlo
for sim in range(N_SIM):
    mat = mat_nom.copy()  # partir des valeurs nominales
    # Pour les critères incertains → tirer une valeur aléatoire dans [min, max]
    for j, c in enumerate(CRITERIA):
        if c in UNCERTAIN:
            mat[:, j] = rng.uniform(mat_min[:, j], mat_max[:, j])
    # Synchronisation : a8 partage les mêmes valeurs que a2, a9 partage les mêmes valeurs que a7
    mat[7,3]=mat[1,3]; mat[8,3]=mat[6,3]  # VED (colonne 3)
    mat[7,6]=mat[1,6]; mat[8,6]=mat[6,6]  # Coût opératoire (colonne 6)
    mc_matrices[sim] = mat  # sauvegarder la matrice
    scores = topsis_score(mat, weights_ahp)  # calculer les scores TOPSIS
    mc_scores[sim] = scores
    mc_ranks[sim]  = N_ALT - scores.argsort().argsort()  # rang 1 = meilleur score

# Normalisation min-max pour les coordonnées parallèles (valeurs entre 0 et 1)
mm_min = mat_min.min(axis=0)   # minimum global par critère
mm_max = mat_max.max(axis=0)   # maximum global par critère
mm_range = mm_max - mm_min
mm_range[mm_range == 0] = 1    # éviter division par zéro si min = max

# Normaliser toutes les valeurs simulées entre 0 et 1
mc_norm = (mc_matrices - mm_min) / mm_range  # résultat : (N_SIM, N_ALT, N_CRIT)

# Inverser les critères de coût pour que 0 = pire et 1 = meilleur dans tous les cas
for j, c in enumerate(CRITERIA):
    if c in COST:
        mc_norm[:,:,j] = 1.0 - mc_norm[:,:,j]

# Moyenne et écart-type des valeurs normalisées (utilisés pour les coordonnées parallèles)
norm_mean = mc_norm.mean(axis=0)  # (N_ALT, N_CRIT) → valeur moyenne normalisée
norm_std  = mc_norm.std(axis=0)   # (N_ALT, N_CRIT) → écart-type normalisé

# ═══════════════════════════════════════════════════════════════════════════════
# CRITÈRES SUPPLÉMENTAIRES c8 / c9 — AFFICHAGE SEULEMENT (pas dans TOPSIS)
# Incertitude ALÉATOIRE (variabilité naturelle du procédé) → distribution NORMALE
# ═══════════════════════════════════════════════════════════════════════════════

# Données [min, max] par alternative (fiches constructeur / mesures procédé)
c8_range = {   # c8 — Surface Roughness Ra (µm) — Min↘
    "a1":(8,16),  "a2":(5,12),  "a3":(5,12),  "a4":(5,12),  "a5":(4,10),
    "a6":(5,13),  "a7":(6,14),  "a8":(5,12),  "a9":(6,14),
}
c9_range = {   # c9 — Porosity (%) — Min↘ (= 100 - Relative Density)
    "a1":(0.5,1.5), "a2":(0.2,1.0), "a3":(0.3,1.2), "a4":(0.5,1.5),
    "a5":(0.1,0.8), "a6":(0.5,1.5), "a7":(0.2,1.0), "a8":(0.2,1.0),
    "a9":(0.2,1.0),
}

N_EXTRA = len(EXTRA_CRITERIA)  # = 2

mat_min_extra = np.zeros((N_ALT, N_EXTRA))
mat_max_extra = np.zeros((N_ALT, N_EXTRA))
for idx, alt in enumerate(ALTS):
    mat_min_extra[idx,0], mat_max_extra[idx,0] = c8_range[alt]
    mat_min_extra[idx,1], mat_max_extra[idx,1] = c9_range[alt]
mat_nom_extra = (mat_min_extra + mat_max_extra) / 2.0

# Monte Carlo — distribution NORMALE : mu = (min+max)/2, sigma = (max-min)/4, clip dans [min, max]
mc_extra = np.zeros((N_SIM, N_ALT, N_EXTRA))
for idx in range(N_ALT):
    for j in range(N_EXTRA):
        mu    = mat_nom_extra[idx, j]
        sigma = (mat_max_extra[idx, j] - mat_min_extra[idx, j]) / 4.0
        mc_extra[:, idx, j] = rng.normal(mu, sigma, N_SIM).clip(mat_min_extra[idx, j], mat_max_extra[idx, j])

# Matrices combinées (7 critères TOPSIS + 2 critères d'affichage) — utilisées par le scatter plot
mat_min_full = np.hstack([mat_min, mat_min_extra])
mat_max_full = np.hstack([mat_max, mat_max_extra])
mat_nom_full = np.hstack([mat_nom, mat_nom_extra])

# Normalisation min-max des critères supplémentaires (pour les coordonnées parallèles)
extra_min_g = mat_min_extra.min(axis=0)
extra_max_g = mat_max_extra.max(axis=0)
extra_range_g = extra_max_g - extra_min_g
extra_range_g[extra_range_g == 0] = 1

mc_extra_norm = (mc_extra - extra_min_g) / extra_range_g
for j, c in enumerate(EXTRA_CRITERIA):
    if c in EXTRA_COST:
        mc_extra_norm[:,:,j] = 1.0 - mc_extra_norm[:,:,j]  # inverser coût (c8) → 1 = meilleur

extra_norm_mean = mc_extra_norm.mean(axis=0)  # (N_ALT, N_EXTRA)
extra_norm_std  = mc_extra_norm.std(axis=0)

# Moyenne/écart-type normalisés pour les 9 critères combinés (coordonnées parallèles)
full_norm_mean = np.hstack([norm_mean, extra_norm_mean])  # (N_ALT, 9)
full_norm_std  = np.hstack([norm_std, extra_norm_std])

# Sous-échantillon fixe pour l'affichage des violin plots (100k points × 9 alt
# est inutilement lourd à transférer/dessiner côté navigateur ; la forme de la
# distribution est déjà bien représentée avec quelques milliers de points).
VIOLIN_N = min(1500, N_SIM)
violin_idx = rng.choice(N_SIM, size=VIOLIN_N, replace=False)

# ═══════════════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def rgba(hex_color, alpha=1.0):
    """Convertit une couleur hexadécimale (#RRGGBB) en format rgba(r,g,b,alpha).
    Utilisée pour rendre les couleurs transparentes (ex: alpha=0.18 = 18% opaque)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ═══════════════════════════════════════════════════════════════════════════════
# STYLES CSS
# ═══════════════════════════════════════════════════════════════════════════════

# Style de carte blanche avec ombre (utilisé avec **CARD pour "déplier" le dictionnaire)
CARD  = {"background":"white","borderRadius":"8px","padding":"16px",
         "boxShadow":"0 1px 4px rgba(0,0,0,0.08)","marginBottom":"12px"}
H3    = {"margin":"0 0 8px","fontSize":"15px","fontWeight":"600","color":"#1a1a2e"}
HINT  = {"margin":"0 0 6px","fontSize":"12px","color":"#aaa"}   # texte gris d'indication
LABEL = {"fontSize":"12px","fontWeight":"bold","marginBottom":"4px","display":"block"}

# Options pour les menus déroulants (critères disponibles, c1–c9)
axis_opts_full  = [{"label":crit_label(c),"value":c} for c in FULL_CRITERIA]

# NOTE : les couleurs/traits épistémique-aléatoire-certain viennent maintenant
# uniquement de COLOR_MAP / DASH_MAP / FILL_MAP (définis en haut du fichier),
# via crit_color(c) / crit_dash(c) / crit_fill(c). Plus de constantes locales.

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION DASH
# ═══════════════════════════════════════════════════════════════════════════════

# Créer l'application Dash
# suppress_callback_exceptions=True → pas d'erreur si un composant n'existe pas encore
# serve_locally=False → charger Plotly.js depuis internet (plus rapide)
app = dash.Dash(__name__, suppress_callback_exceptions=True, serve_locally=False,
                external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "MACCOD — Metal 3D Printer Selection"

# ── Textes d'aide (bouton "?" à côté de chaque titre de visualisation) ──────
INFO_TEXTS = {
    "parcoords": "Ce graphe montre toutes les imprimantes sur tous les critères "
                 "en même temps. Chaque ligne colorée = une imprimante. Les bandes "
                 "autour des lignes montrent l'incertitude.",
    "scatter": "Ce graphe compare deux critères à la fois. Chaque point = une "
               "imprimante. Les barres verticales et horizontales montrent "
               "l'intervalle d'incertitude.",
    "violin": "Ce graphe montre comment les valeurs peuvent varier pour chaque "
              "imprimante. Plus le violon est large, plus l'incertitude est "
              "grande.",
    "matrix": "Ce tableau compare toutes les imprimantes sur tous les critères. "
              "La longueur de la barre = la performance. La partie dégradée = "
              "l'incertitude.",
    "heatmap": "Ce tableau montre la probabilité que chaque imprimante obtienne "
               "chaque rang. Une case foncée = très probable. Utilisez-le pour "
               "voir si le classement est stable.",
}

def info_button(key):
    """Petit bouton "?" qui ouvre la modale d'explication associée à `key`."""
    return dbc.Button("?", id=f"info-btn-{key}", size="sm", color="secondary", outline=True,
                       style={"borderRadius":"50%","width":"22px","height":"22px","padding":"0",
                              "fontSize":"12px","lineHeight":"1","marginLeft":"8px"})

def info_modal(key, title_text):
    """Popover d'explication ancré au bouton "?" (voir INFO_TEXTS) : s'ouvre/se
    ferme au clic sur le bouton, toujours à l'emplacement du graphe concerné
    (contrairement à un dbc.Modal, centré dans toute la fenêtre)."""
    return dbc.Popover([
        dbc.PopoverHeader(title_text),
        dbc.PopoverBody(INFO_TEXTS[key]),
    ], target=f"info-btn-{key}", trigger="click", placement="bottom",
       style={"maxWidth":"320px"})

# Layout = la structure HTML de la page (ce que l'utilisateur voit)
app.layout = html.Div([

    # ── En-tête ──────────────────────────────────────────────────────────────
    # Titre principal et description du projet
    html.Div([
        html.H2("Metal 3D Printer Selection — MCDM Dashboard",
                style={"margin":"0","color":"#1a1a2e","fontSize":"20px"}),
        html.P("Projet ANR MACCOD | LURPA, ENS Paris-Saclay  ·  "
               "9 alternatives × 9 critères (7 TOPSIS + 2 affichage) × Monte Carlo (100 000 simulations)",
               style={"margin":"4px 0 0","color":"#666","fontSize":"13px"}),
        html.P("TOPSIS (c1–c7) : c4 (VED), c6 (Coût acquisition), c7 (Coût opératoire) ◆ épistémique "
               "— uniforme dans [min, max]. Affichage seulement (c8–c9) : c8 (Rugosité Ra), "
               "c9 (Porosité) ● aléatoire — normale, non utilisés dans le classement TOPSIS.",
               style={"margin":"2px 0 0","color":"#999","fontSize":"11px"}),
    ], style=CARD),

    # ── Légende globale — encodage incertitude ───────────────────────────────
    html.Div([
        html.Span("◆ Incertitude épistémique (uniforme) — données constructeur — réductible",
                  style={"background":FILL_MAP["epistemic"],
                         "border":f'1px {CSS_BORDER_STYLE[DASH_MAP["epistemic"]]} {COLOR_MAP["epistemic"]}',
                         "borderRadius":"4px","padding":"5px 12px","marginRight":"14px",
                         "fontSize":"12px","color":"#1a3d6b","display":"inline-block"}),
        html.Span("● Incertitude aléatoire (normale) — variabilité procédé — irréductible",
                  style={"background":FILL_MAP["aleatoric"],
                         "border":f'1px {CSS_BORDER_STYLE[DASH_MAP["aleatoric"]]} {COLOR_MAP["aleatoric"]}',
                         "borderRadius":"4px","padding":"5px 12px",
                         "fontSize":"12px","color":"#7a3900","display":"inline-block"}),
    ], style={**CARD,"padding":"10px 16px"}),

    # ── Barre de contrôle ────────────────────────────────────────────────────
    # Menus pour choisir les axes du scatter + checkbox incertitude + bouton reset
    html.Div([
        html.Div([
            html.Span("Axe X (scatter)", style=LABEL),
            # Menu déroulant pour choisir le critère de l'axe X (c1–c9)
            dcc.Dropdown(axis_opts_full,"c6",id="x-axis",clearable=False,style={"width":"220px"}),
        ]),
        html.Div([
            html.Span("Axe Y (scatter)", style=LABEL),
            # Menu déroulant pour choisir le critère de l'axe Y (c1–c9)
            dcc.Dropdown(axis_opts_full,"c4",id="y-axis",clearable=False,style={"width":"220px"}),
        ]),
        html.Div([
            html.Span("Incertitude (coord. parallèles)", style=LABEL),
            # Case à cocher : afficher ou non les bandes d'incertitude Monte Carlo
            dcc.Checklist(
                options=[{"label":"  Bandes Monte Carlo (µ ± σ)","value":"show"}],
                value=["show"], id="show-uncertainty",  # value=["show"] → case cochée par défaut
                style={"fontSize":"13px","marginTop":"6px"},
            ),
        ]),
        html.Div([
            # Bouton pour réinitialiser la sélection (effacer tous les points sélectionnés)
            html.Button("↺ Réinitialiser", id="reset-btn", n_clicks=0,
                style={"marginTop":"18px","padding":"8px 16px","background":"#4472C4",
                       "color":"white","border":"none","borderRadius":"4px",
                       "cursor":"pointer","fontSize":"13px"}),
        ]),
    ], style={**CARD,"display":"flex","gap":"20px","alignItems":"flex-start","flexWrap":"wrap"}),

    # ── Stockage partagé + Barre de sélection ────────────────────────────────
    # Affiche les alternatives sélectionnées sous forme d'étiquettes colorées
    html.Div(id="selection-bar",
             style={"marginBottom":"12px","fontSize":"13px","color":"#555","minHeight":"24px"}),
    # dcc.Store = mémoire invisible partagée entre tous les graphes
    # Stocke la liste des alternatives sélectionnées (ex: ["a1","a3"])
    dcc.Store(id="selected-alts", data=[]),

    # ── Coordonnées parallèles ────────────────────────────────────────────────
    # Graphe principal : une ligne par alternative, un axe par critère (c1–c9)
    html.Div([
        html.Div([
            html.H3(id="parcoords-title", style={**H3,"margin":"0"}),  # titre mis à jour par callback
            info_button("parcoords"),
        ], style={"display":"flex","alignItems":"center","marginBottom":"8px"}),
        dcc.Graph(id="parcoords-graph", figure=go.Figure(),
                  style={"height":"440px"}, config={"displayModeBar":"hover"}),
        html.P("Bande continue = épistémique · Bande pointillée = aléatoire",
               style={**HINT,"marginTop":"6px","marginBottom":"0"}),
        info_modal("parcoords", "Coordonnées parallèles"),
    ], style=CARD),

    # ── Scatter + Violin — côte à côte ───────────────────────────────────────
    html.Div([
        # Scatter
        html.Div([
            html.Div([
                html.H3("Nuage de points — Barres d'erreur (intervalles)", style={**H3,"margin":"0"}),
                info_button("scatter"),
            ], style={"display":"flex","alignItems":"center","marginBottom":"8px"}),
            html.P("Sélectionnez des points (rectangle ou lasso).", style=HINT),
            dcc.Graph(id="scatter-graph", figure=go.Figure(), style={"height":"420px"},
                      config={"displayModeBar":"hover"}),
            html.P("Bleu = épistémique · Orange = aléatoire",
                   style={**HINT,"marginTop":"6px","marginBottom":"0"}),
            info_modal("scatter", "Nuage de points"),
        ], style={**CARD,"flex":"1","minWidth":"0"}),

        # Violin — un seul graphe, critère choisi parmi c1–c9
        # Le style (bleu plein ◆ / orange pointillé ● / gris = certain)
        # s'adapte automatiquement selon le type d'incertitude du critère choisi.
        html.Div([
            html.Div([
                html.H3(id="violin-title", style={**H3,"margin":"0"}),
                info_button("violin"),
            ], style={"display":"flex","alignItems":"center","marginBottom":"8px"}),
            html.Div([
                dcc.Dropdown(axis_opts_full,"c4",id="violin-crit",clearable=False,
                             style={"width":"220px"}),
            ], style={"marginBottom":"8px"}),
            dcc.Graph(id="violin-graph", figure=go.Figure(), style={"height":"400px"},
                      config={"displayModeBar":"hover"}),
            html.P("Bleu = épistémique · Orange = aléatoire · Gris = certain",
                   style={**HINT,"marginTop":"6px","marginBottom":"0"}),
            info_modal("violin", "Diagramme en violon"),
        ], style={**CARD,"flex":"1","minWidth":"0"}),
    ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),

    # ── Matrice de comparaison — Alternatives × Critères ─────────────────────
    html.Div([
        html.Div([
            html.H3("Matrice de comparaison — Alternatives × Critères", style={**H3,"margin":"0"}),
            info_button("matrix"),
        ], style={"display":"flex","alignItems":"center","marginBottom":"8px"}),
        html.Div(id="matrix-graph"),
        info_modal("matrix", "Matrice de comparaison"),
    ], style=CARD),

    # ── Classement probabiliste ──────────────────────────────────────────────
    html.Div([
        html.H3("Classement probabiliste — Monte Carlo TOPSIS", style=H3),
        # Section repliable pour ajuster les poids des critères
        html.Details([
            html.Summary("Ajuster les pondérations des critères",
                         style={"fontSize":"13px","cursor":"pointer","color":"#4472C4",
                                "marginBottom":"10px"}),
            # Créer un champ numérique pour chaque critère (poids ajustables)
            html.Div([
                html.Div([
                    html.Span(f"{c} ({SHORT[c]})", style={"fontSize":"12px","fontWeight":"bold",
                                             "display":"block","marginBottom":"2px"}),
                    # Champ numérique avec valeur par défaut = poids AHP
                    dcc.Input(id=f"w-{c}", type="number", min=0, max=10, step=0.01,
                              value=w_val,
                              style={"width":"80px","padding":"4px 8px","fontSize":"13px",
                                     "border":"1px solid #ccc","borderRadius":"4px"}),
                ], style={"display":"inline-block","marginRight":"20px","marginBottom":"8px"})
                for c, w_val in zip(CRITERIA, weights_ahp)  # créer un champ par critère
            ]),
        ], style={"marginBottom":"16px"}),
        # Zone où le tableau de classement + la heatmap P(rang) seront affichés
        html.Div([
            html.Div(id="recommendation-text", style={"flex":"1"}),
            html.Div([
                html.Div([
                    html.H4("Heatmap P(rang)", style={**H3,"fontSize":"14px","margin":"0"}),
                    info_button("heatmap"),
                ], style={"display":"flex","alignItems":"center","marginBottom":"6px"}),
                dcc.Graph(id="heatmap-rang", style={"height":"380px"}),
            ], style={"flex":"1"}),
        ], style={"display":"flex","gap":"24px","alignItems":"flex-start"}),
        info_modal("heatmap", "Heatmap P(rang)"),
    ], style=CARD),

], style={"fontFamily":"Arial, sans-serif","maxWidth":"1440px","margin":"0 auto",
          "padding":"16px","background":"#f5f7fa","minHeight":"100vh"})

# ═══════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Callback 1 : Sélection rectangle/lasso dans le scatter ───────────────────
# Déclenché quand l'utilisateur sélectionne des points dans le scatter plot
# → extrait les noms des alternatives sélectionnées et les stocke
@callback(
    Output("selected-alts","data", allow_duplicate=True),
    Input("scatter-graph","selectedData"),  # sélection rectangle ou lasso
    prevent_initial_call=True,  # ne pas s'exécuter au démarrage
)
def scatter_to_store(sel):
    if not sel or not sel.get("points"):
        return dash.no_update  # ne pas écraser la sélection lors du redessin
    # Extraire le nom de chaque point sélectionné (customdata[0] = nom de l'alternative)
    # set {...} = pas de doublons, list(...) = convertir en liste
    return list({p["customdata"][0] for p in sel["points"] if p.get("customdata")})

# ── Callback 2 : Bouton Reset ─────────────────────────────────────────────────
# Déclenché quand l'utilisateur clique sur "↺ Réinitialiser"
# → vide la sélection (tous les graphes reviennent à l'état normal)
@callback(
    Output("selected-alts","data", allow_duplicate=True),
    Input("reset-btn","n_clicks"),  # nombre de clics sur le bouton
    prevent_initial_call=True,
)
def reset_store(_):
    # _ = on reçoit n_clicks mais on ne l'utilise pas (peu importe le nombre de clics)
    return []  # toujours retourner liste vide → effacer la sélection

# ── Callback 3 : Clic dans les coordonnées parallèles ────────────────────────
# Déclenché quand l'utilisateur clique sur une ligne dans le graphe parcoords
# → ajoute ou enlève cette alternative de la sélection (ON/OFF)
@callback(
    Output("selected-alts","data", allow_duplicate=True),
    Input("parcoords-graph","clickData"),   # déclenche le callback
    State("selected-alts","data"),          # lit la sélection actuelle (sans déclencher)
    prevent_initial_call=True,
)
def parcoords_click_to_store(click, selected):
    # Vérification de sécurité : si rien cliqué → ne rien changer
    if not click or not click.get("points"):
        return selected or []
    pt = click["points"][0]   # premier point cliqué
    cd = pt.get("customdata") # customdata = nom de l'alternative qu'on a mis nous-mêmes
    if not cd:
        return selected or []
    alt = cd[0]  # nom de l'alternative (ex: "a3")
    # Vérifier que c'est bien une de nos alternatives
    if alt not in ALTS:
        return selected or []
    sel = list(selected or [])  # copie de la sélection actuelle (ou liste vide)
    # Logique ON/OFF : clic = ajouter, second clic = enlever
    if alt in sel:
        sel.remove(alt)  # déjà sélectionnée → désélectionner
    else:
        sel.append(alt)  # pas encore sélectionnée → ajouter
    return sel

# ── Callback 5 : Clic dans le violin plot ────────────────────────────────────
# Identique au box plot — utilise aussi curveNumber pour identifier l'alternative
@callback(
    Output("selected-alts","data", allow_duplicate=True),
    Input("violin-graph","clickData"),
    State("selected-alts","data"),
    prevent_initial_call=True,
)
def violin_click_to_store(click, selected):
    if not click or not click.get("points"):
        return selected or []
    curve = click["points"][0].get("curveNumber", -1)
    alts_show = list(selected) if selected else list(ALTS)
    if curve < 0 or curve >= len(alts_show):
        return selected or []
    alt = alts_show[curve]
    sel = list(selected or [])
    if alt in sel:
        sel.remove(alt)
    else:
        sel.append(alt)
    return sel

# ── Callback 6 : Barre de sélection ──────────────────────────────────────────
# Déclenché quand la sélection change → met à jour le texte en haut avec les étiquettes colorées
@callback(Output("selection-bar","children"), Input("selected-alts","data"))
def update_selection_bar(selected):
    if not selected:
        # Rien sélectionné → afficher le texte par défaut en gris
        return html.Span("Sélection : toutes les alternatives (9/9)", style={"color":"#999"})
    # Créer une étiquette colorée pour chaque alternative sélectionnée
    chips = [html.Span(a, style={
        "display":"inline-block","background":ALT_COLOR.get(a,"#999"),
        "color":"white","borderRadius":"12px","padding":"2px 10px",
        "fontSize":"12px","marginRight":"6px","fontWeight":"bold",
    }) for a in sorted(selected)]  # sorted = trier par ordre alphabétique
    return html.Span([
        html.Span(f"Sélection ({len(selected)}/9) : ",
                  style={"fontWeight":"bold","marginRight":"6px"}), *chips])

# ── Callback 7 : Titre des coordonnées parallèles ────────────────────────────
# Déclenché quand la checkbox change → met à jour le titre du graphe
@callback(Output("parcoords-title","children"), Input("show-uncertainty","value"))
def update_parcoords_title(show):
    if show:  # case cochée → show = ["show"]
        return "Coordonnées parallèles"
    return "Coordonnées parallèles — Valeurs nominales"  # case décochée → show = []

# ── Callback 8 : Dessiner les coordonnées parallèles ─────────────────────────
# Déclenché quand la sélection ou la checkbox change
# → redessine entièrement le graphe des coordonnées parallèles
@callback(
    Output("parcoords-graph","figure"),
    Input("selected-alts","data"),       # sélection actuelle
    Input("show-uncertainty","value"),   # checkbox incertitude
)
def render_parcoords(selected, show_unc):
    show_unc = bool(show_unc)   # convertir ["show"] ou [] en True/False
    fig = go.Figure()           # créer une figure vide
    n_axes = len(FULL_CRITERIA)  # 9 axes : c1..c7 (TOPSIS) + c8, c9 (affichage)
    x = list(range(n_axes))     # positions horizontales des axes : [0,1,...,8]

    alts_sel = selected if selected else []  # alternatives sélectionnées
    all_shown = not bool(selected)           # True = tout afficher (rien sélectionné)

    # ── Étape 1 : Dessiner les axes verticaux gris ───────────────────────────
    # Une ligne verticale grise par critère (derrière les données)
    for xi in x:
        fig.add_shape(type="line", x0=xi, x1=xi, y0=0, y1=1,
                      line=dict(color="#d0d0d0", width=1), layer="below")

    # ── Étape 2 : Bandes d'incertitude (seulement si checkbox cochée) ────────
    # Ruban par alternative, coloré par ALT_COLOR, qui suit la moyenne ±
    # écart-type sur c4,c6,c7 (épistémique) et c8,c9 (aléatoire). La bande
    # n'est dessinée QUE sur les tronçons reliant deux axes incertains
    # consécutifs (jamais en direction d'un axe certain), pour ne pas donner
    # l'illusion visuelle d'une incertitude sur c1,c2,c3,c5 (qui n'en ont
    # aucune) : le "rétrécissement" de la bande vers ces axes créait cette
    # confusion. Un axe incertain isolé entre deux axes certains (ex. c4)
    # reçoit à la place un petit repère vertical (moustache), sans bande.
    # La bordure du ruban devient pointillée exactement sur le tronçon
    # aléatoire (c8-c9), pour signaler visuellement ce changement de nature
    # d'incertitude.
    if show_unc:
        is_uncertain_axis = [crit_type(c) != "certain" for c in FULL_CRITERIA]
        # Regrouper les axes incertains consécutifs en "runs" (ex. [3], [5,6,7,8])
        runs, cur = [], []
        for j, unc in enumerate(is_uncertain_axis):
            if unc:
                cur.append(j)
            elif cur:
                runs.append(cur); cur = []
        if cur:
            runs.append(cur)

        for idx, alt in enumerate(ALTS):  # idx = numéro, alt = nom de l'alternative
            is_sel = all_shown or alt in alts_sel  # cette alternative est-elle sélectionnée ?
            if not is_sel:
                continue  # sauter les alternatives non sélectionnées
            color = ALT_COLOR[alt]
            lo_v = []  # valeurs basses de la bande (µ - σ)
            hi_v = []  # valeurs hautes de la bande (µ + σ)
            for j, c in enumerate(FULL_CRITERIA):
                t = crit_type(c)
                if t == "epistemic":
                    lo_v.append(max(0.0, norm_mean[idx, j] - norm_std[idx, j]))
                    hi_v.append(min(1.0, norm_mean[idx, j] + norm_std[idx, j]))
                elif t == "aleatoric":
                    je = EXTRA_CRITERIA.index(c)
                    lo_v.append(max(0.0, extra_norm_mean[idx, je] - extra_norm_std[idx, je]))
                    hi_v.append(min(1.0, extra_norm_mean[idx, je] + extra_norm_std[idx, je]))
                else:
                    # Critère certain → pas de bande (lo = hi = même valeur)
                    lo_v.append(full_norm_mean[idx, j])
                    hi_v.append(full_norm_mean[idx, j])

            for run in runs:
                if len(run) == 1:
                    # Axe incertain isolé (ex. c4) → petit repère vertical, pas de bande
                    j0 = run[0]
                    fig.add_trace(go.Scatter(
                        x=[x[j0], x[j0]], y=[lo_v[j0], hi_v[j0]], mode="lines",
                        line=dict(color=color, width=4), opacity=0.35,
                        legendgroup=alt, showlegend=False, hoverinfo="skip",
                    ))
                    continue
                xs = [x[j] for j in run]
                los = [lo_v[j] for j in run]
                his = [hi_v[j] for j in run]
                # Trace 1 : ligne basse invisible (sert de plancher pour le remplissage)
                fig.add_trace(go.Scatter(
                    x=xs, y=los, mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),  # totalement invisible
                    legendgroup=alt, showlegend=False, hoverinfo="skip",
                ))
                # Trace 2 : ligne haute invisible + remplissage jusqu'à la trace précédente
                fig.add_trace(go.Scatter(
                    x=xs, y=his, mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),  # totalement invisible
                    fill="tonexty",               # remplir entre cette ligne et la précédente
                    fillcolor=rgba(color, 0.18),  # couleur de l'alternative à 18% d'opacité
                    legendgroup=alt, showlegend=False, hoverinfo="skip",
                ))

                # Surcouche pointillée sur le tronçon aléatoire (c8-c9) uniquement,
                # s'il fait partie de ce run
                ale_in_run = [j for j in run if crit_type(FULL_CRITERIA[j]) == "aleatoric"]
                if len(ale_in_run) >= 2:
                    jf, jl = ale_in_run[0], ale_in_run[-1]
                    fig.add_trace(go.Scatter(
                        x=[x[jf], x[jl]], y=[hi_v[jf], hi_v[jl]], mode="lines",
                        line=dict(color=color, width=1.5, dash="dash"),
                        legendgroup=alt, showlegend=False, hoverinfo="skip",
                    ))
                    fig.add_trace(go.Scatter(
                        x=[x[jf], x[jl]], y=[lo_v[jf], lo_v[jl]], mode="lines",
                        line=dict(color=color, width=1.5, dash="dash"),
                        legendgroup=alt, showlegend=False, hoverinfo="skip",
                    ))

    # ── Étape 3 : Lignes nominales (valeurs moyennes) ────────────────────────
    # Dessine une ligne colorée par alternative (épaisse si sélectionnée, grise sinon)
    for idx, alt in enumerate(ALTS):
        is_sel = all_shown or alt in alts_sel
        # Style selon si l'alternative est sélectionnée ou non
        color = ALT_COLOR[alt] if is_sel else "#d5d5d5"  # couleur ou gris
        lw    = 2.5 if is_sel else 1.0   # épaisseur de ligne
        op    = 1.0 if is_sel else 0.4   # opacité (transparence)
        nom_vals = full_norm_mean[idx].tolist()  # valeurs normalisées moyennes de cette alternative
        # Afficher le nom de l'alternative seulement à droite (dernier axe)
        labels = [""] * (n_axes - 1) + [f"  {alt}"]

        fig.add_trace(go.Scatter(
            x=x, y=nom_vals,
            mode="lines+text",  # ligne + texte, sans marqueurs de points
            text=labels, textposition="middle right",
            textfont=dict(size=10, color=color),
            name=f"{alt}",              # nom dans la légende
            legendgroup=alt,            # groupe pour cacher/montrer toutes les traces d'une alternative
            showlegend=is_sel,          # apparaît dans la légende seulement si sélectionnée
            opacity=op,
            line=dict(color=color, width=lw),
            customdata=[[alt]]*n_axes,  # nom de l'alternative attaché à chaque point (utilisé par callbacks)
            hovertemplate=f"<b>{alt}</b><br>"
                          + "<br>".join(f"{SHORT[c]}: %{{y:.3f}}" if j==0
                                        else "" for j, c in enumerate(FULL_CRITERIA))
                          + "<extra></extra>",
        ))

    # ── Mise en page finale ───────────────────────────────────────────────────
    fig.update_layout(
        xaxis=dict(
            tickmode="array", tickvals=x,
            # Étiquettes des axes : la couleur (gris/bleu/orange) suffit à coder
            # le type — pas de symbole ◆/● supplémentaire ici (redondant)
            ticktext=[f'<span style="color:{crit_color(c)}"><b>{c}</b><br>{SHORT[c]}</span>'
                      for c in FULL_CRITERIA],
            # Padding symétrique autour des axes (0..n_axes-1) pour centrer la grille
            tickangle=0, showgrid=False, range=[-0.5, n_axes - 1 + 0.5],
        ),
        yaxis=dict(
            range=[-0.05, 1.08], title="Valeur normalisée [0–1]",
            showgrid=True, gridcolor="#f0f0f0",
            zeroline=True, zerolinecolor="#ccc",
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),  # légende horizontale au-dessus
        # Marges quasi symétriques (gauche = titre axe Y, droite = juste la place
        # du nom d'alternative en bout de ligne) pour centrer la grille dans la card
        margin=dict(l=60, r=70, t=40, b=70),
        hovermode="closest",
    )
    return fig

# ── Callback 9 : Dessiner le scatter plot ────────────────────────────────────
# Déclenché quand la sélection ou les axes X/Y changent
# → redessine le nuage de points avec barres d'erreur
@callback(
    Output("scatter-graph","figure"),
    Input("selected-alts","data"),   # sélection actuelle
    Input("x-axis","value"),         # critère choisi pour l'axe X
    Input("y-axis","value"),         # critère choisi pour l'axe Y
)
def render_scatter(selected, x_col, y_col):
    fig = go.Figure()
    jx = FULL_CRITERIA.index(x_col)  # position du critère X dans la liste (ex: "c8" → 7)
    jy = FULL_CRITERIA.index(y_col)  # position du critère Y dans la liste

    # Barres d'erreur natives (Plotly ne permet pas de pointillés sur error_x/error_y) :
    # ÉPISTÉMIQUE (c4,c6,c7) → barre native, trait continu bleu (DASH_MAP="solid" le permet).
    # ALÉATOIRE (c8,c9) → barre native désactivée, remplacée par un segment manuel
    # en pointillé orange (Plotly ne sait pas nativement tracer un error bar en dash).
    # CERTAIN (c1,c2,c3,c5) → aucune barre d'erreur (pas d'incertitude à montrer).
    # Le fondu (sélection/non-sélection) est géré par une SEULE opacité de trace,
    # appliquée uniformément au marqueur, aux barres natives ET aux segments manuels
    # (sinon les barres/segments restaient pleinement visibles même hors sélection).

    for idx, alt in enumerate(ALTS):
        hi = not selected or alt in selected  # hi = "highlighted" = est-ce sélectionnée ?
        color = ALT_COLOR[alt]
        fade = 1.0 if hi else 0.2  # opacité commune : sélectionné = plein, sinon estompé

        # Valeurs nominales (centre du point)
        x_nom = float(mat_nom_full[idx, jx])
        y_nom = float(mat_nom_full[idx, jy])

        # Barres d'erreur = différence entre nominal et [min, max]
        ex_p = float(mat_max_full[idx, jx] - mat_nom_full[idx, jx])  # barre à droite (+)
        ex_m = float(mat_nom_full[idx, jx] - mat_min_full[idx, jx])  # barre à gauche (-)
        ey_p = float(mat_max_full[idx, jy] - mat_nom_full[idx, jy])  # barre en haut (+)
        ey_m = float(mat_nom_full[idx, jy] - mat_min_full[idx, jy])  # barre en bas (-)

        x_type = crit_type(x_col)
        y_type = crit_type(y_col)
        x_ale, y_ale = x_type == "aleatoric", y_type == "aleatoric"

        # ── Segments manuels pointillés oranges pour les axes aléatoires ─────
        ale_color = COLOR_MAP["aleatoric"]
        cap_half = 6  # demi-longueur du cap en unités de pixels (via un petit delta en données)
        if x_ale:
            x_lo, x_hi = x_nom - ex_m, x_nom + ex_p
            fig.add_trace(go.Scatter(
                x=[x_lo, x_hi], y=[y_nom, y_nom], mode="lines",
                line=dict(color=ale_color, width=2, dash="dot"), opacity=fade,
                legendgroup=alt, showlegend=False, hoverinfo="skip",
            ))
            # Caps verticaux aux extrémités (en utilisant des marqueurs line-ns)
            fig.add_trace(go.Scatter(
                x=[x_lo, x_hi], y=[y_nom, y_nom], mode="markers",
                marker=dict(symbol="line-ns", size=cap_half*2, color=ale_color,
                            line=dict(width=2, color=ale_color)), opacity=fade,
                legendgroup=alt, showlegend=False, hoverinfo="skip",
            ))
        if y_ale:
            y_lo, y_hi = y_nom - ey_m, y_nom + ey_p
            fig.add_trace(go.Scatter(
                x=[x_nom, x_nom], y=[y_lo, y_hi], mode="lines",
                line=dict(color=ale_color, width=2, dash="dot"), opacity=fade,
                legendgroup=alt, showlegend=False, hoverinfo="skip",
            ))
            # Caps horizontaux aux extrémités
            fig.add_trace(go.Scatter(
                x=[x_nom, x_nom], y=[y_lo, y_hi], mode="markers",
                marker=dict(symbol="line-ew", size=cap_half*2, color=ale_color,
                            line=dict(width=2, color=ale_color)), opacity=fade,
                legendgroup=alt, showlegend=False, hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=[x_nom], y=[y_nom],
            mode="markers+text",
            text=[alt], textposition="top center",  # nom au-dessus du point
            # customdata stocke des infos supplémentaires pour le hovertemplate
            customdata=[[alt, x_nom-ex_m, x_nom+ex_p, y_nom-ey_m, y_nom+ey_p]],
            name=alt, showlegend=True, legendgroup=alt,
            opacity=fade,  # fondu uniforme : marqueur + barres d'erreur natives
            marker=dict(size=14 if hi else 8, color=color,
                        line=dict(width=2 if hi else 0, color="black")),  # bordure noire si sélectionné
            # Barres d'erreur natives (visibles seulement pour un axe épistémique —
            # aléatoire = segments manuels ci-dessus, certain = pas de barre du tout)
            error_x=dict(type="data", symmetric=False,
                         array=[ex_p], arrayminus=[ex_m], visible=(x_type == "epistemic"),
                         color=COLOR_MAP["epistemic"], thickness=1.5, width=6),
            error_y=dict(type="data", symmetric=False,
                         array=[ey_p], arrayminus=[ey_m], visible=(y_type == "epistemic"),
                         color=COLOR_MAP["epistemic"], thickness=1.5, width=6),
            # Info-bulle au survol : nom + valeur + intervalle [min-max]
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{SHORT[x_col]}: %{{x:.2f}} "
                "[%{customdata[1]:.2f} – %{customdata[2]:.2f}]<br>"
                f"{SHORT[y_col]}: %{{y:.2f}} "
                "[%{customdata[3]:.2f} – %{customdata[4]:.2f}]"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis_title=LABELS[x_col], yaxis_title=LABELS[y_col],
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=60, b=60),
        dragmode="select",         # outil rectangle activé par défaut
        clickmode="event+select",  # active les clics ET la sélection rectangle
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig

# ── Callback 10 : Titre du violin plot ───────────────────────────────────────
@callback(Output("violin-title","children"), Input("violin-crit","value"))
def update_violin_title(crit):
    return f"Diagramme en violon — {crit} — {SHORT[crit]}{crit_marker(crit)}"

# ── Callback 11 : Dessiner le violin plot (c1–c9, un seul graphe) ────────────
# Le style s'adapte au type d'incertitude du critère sélectionné :
#   - épistémique (c4,c6,c7) : bleu uniforme + bordure CONTINUE
#   - aléatoire   (c8,c9)    : orange uniforme + bordure POINTILLÉE
#   - certain     (c1,c2,c3,c5) : gris uniforme (pas de couleur dédiée)
@callback(
    Output("violin-graph","figure"),
    Input("selected-alts","data"),
    Input("violin-crit","value"),
)
def render_violin(selected, crit):
    alts_show = selected if selected else ALTS
    t = crit_type(crit)

    if t == "aleatoric":
        j = EXTRA_CRITERIA.index(crit)
        data_source = mc_extra
    else:
        j = CRITERIA.index(crit)
        data_source = mc_matrices

    # Ici, le type d'incertitude est encodé par la couleur seule (pas de
    # pointillé) : un contour recalculé à la main (KDE) déforme légèrement
    # la vraie silhouette du violon par rapport au rendu natif de Plotly.
    fig = go.Figure()
    for idx, alt in enumerate(ALTS):
        if alt not in alts_show:
            continue
        vals = data_source[violin_idx, idx, j]  # sous-échantillon (forme de distribution)
        fig.add_trace(go.Violin(
            y=vals, name=alt,
            box_visible=True,       # afficher la boîte à moustaches à l'intérieur
            meanline_visible=True,  # afficher la ligne de moyenne
            fillcolor=crit_fill(crit),
            line=dict(color=crit_color(crit), width=2),
            opacity=0.9, points=False,  # ne pas afficher les points individuels (trop nombreux)
        ))

    fig.update_layout(
        yaxis_title=LABELS[crit],
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=60, r=20, t=40, b=60),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig

# ── Callback 11bis : Matrice de comparaison Alternatives × Critères ──────────
# Une barre par cellule : partie pleine = valeur nominale normalisée,
# dégradé = extension d'incertitude (µ ± σ). Couleur/bordure selon le type
# d'incertitude du critère (épistémique ◆ bleu continu / aléatoire ● orange
# pointillé / certain gris).
def _matrix_crit_style(c):
    """Retourne (couleur_barre, couleur_remplissage_degrade, bordure_css, fond_entete)
    d'après CRITERION_TYPE/COLOR_MAP/DASH_MAP/FILL_MAP (source unique de vérité)."""
    t = crit_type(c)
    if t == "certain":
        return COLOR_MAP[t], FILL_MAP[t], "solid #ccc", "#f0f4fa"
    css_style = CSS_BORDER_STYLE[DASH_MAP[t]]
    return COLOR_MAP[t], FILL_MAP[t], f"{css_style} {COLOR_MAP[t]}", FILL_MAP[t]

@callback(Output("matrix-graph","children"), Input("selected-alts","data"))
def render_matrix(selected):
    has_sel = bool(selected)

    # ── Clustering hiérarchique des alternatives (réordonne les lignes) ──────
    # Sur la matrice normalisée (9 alt × 9 critères) déjà calculée ailleurs
    # dans l'app (full_norm_mean), pour regrouper les imprimantes similaires.
    Z = linkage(full_norm_mean, method="ward", metric="euclidean")
    order = leaves_list(Z)                          # ordre d'affichage des lignes
    cluster_of = fcluster(Z, t=4, criterion="maxclust")  # 4 groupes fixes, par alternative (index original)

    # ── En-tête ───────────────────────────────────────────────────────────────
    header_cells = [html.Th("Alt.", style={
        "padding":"6px 10px","fontSize":"12px","textAlign":"left","color":"#888",
        "borderBottom":"2px solid #ddd",
    })]
    for c in FULL_CRITERIA:
        _, _, _, head_bg = _matrix_crit_style(c)
        header_cells.append(html.Th([
            html.Div(f"{c}{crit_marker(c)}", style={"fontWeight":"bold"}),
            html.Div(SHORT[c], style={"fontWeight":"normal","fontSize":"10px","color":"#666"}),
        ], style={
            "padding":"6px 8px","fontSize":"12px","textAlign":"center",
            "background":head_bg,"borderBottom":"2px solid #ddd","minWidth":"90px",
        }))

    # ── Lignes (une par alternative, dans l'ordre du clustering) ─────────────
    rows = []
    n_cols = len(FULL_CRITERIA) + 1
    prev_cluster = None
    group_num = 0
    for pos, idx in enumerate(order):
        alt = ALTS[idx]
        cur_cluster = cluster_of[idx]
        if cur_cluster != prev_cluster:
            group_num += 1
            # Séparateur fin entre deux groupes détectés (pas avant le tout premier)
            label_style = {"padding":"4px 10px","fontSize":"11px","color":"#999",
                            "background":"transparent"}
            if pos > 0:
                label_style["borderTop"] = "2px solid #ccc"
            rows.append(html.Tr(html.Td(f"Groupe {group_num}", colSpan=n_cols,
                                         style=label_style)))
            prev_cluster = cur_cluster

        is_sel = (not has_sel) or (alt in selected)
        if has_sel and alt in selected:
            row_bg = "#f0f4ff"
        elif has_sel:  # sélection non vide et cette alt n'en fait pas partie
            row_bg = "#fafafa"
        else:
            row_bg = "transparent"
        row_opacity = "1" if is_sel else "0.45"

        cells = [html.Td(alt, style={
            "padding":"6px 10px","fontWeight":"bold","color":ALT_COLOR[alt],
            "whiteSpace":"nowrap",
        })]

        for j, c in enumerate(FULL_CRITERIA):
            bar_color_, fill_color_, border_css, _ = _matrix_crit_style(c)
            is_uncertain_crit = crit_type(c) != "certain"  # gradient réservé à c4,c6,c7,c8,c9
            mean_pct = float(full_norm_mean[idx, j]) * 100
            std_pct  = float(full_norm_std[idx, j]) * 100
            mean_pct = max(0.0, min(100.0, mean_pct))
            grad_w   = max(0.0, min(std_pct, 100.0 - mean_pct)) if is_uncertain_crit else 0.0

            nominal_val = float(mat_nom_full[idx, j])
            tooltip = f"{alt} — {LABELS[c]} : {nominal_val:.2f} (µ={mean_pct:.0f}% ± σ={std_pct:.0f}%)"

            bar_children = [
                # Barre pleine = valeur nominale normalisée
                html.Div(style={
                    "position":"absolute","left":"0","top":"0","bottom":"0",
                    "width":f"{mean_pct}%","background":bar_color_,
                    "borderRadius":"2px",
                }),
            ]
            if is_uncertain_crit:
                # Dégradé = extension d'incertitude (µ ± σ) — uniquement pour c4,c6,c7,c8,c9
                bar_children.append(html.Div(style={
                    "position":"absolute","top":"0","bottom":"0",
                    "left":f"{mean_pct}%","width":f"{grad_w}%",
                    "background":f"linear-gradient(to right, {rgba(bar_color_,0.35)}, {fill_color_})",
                    "borderRadius":"0 2px 2px 0",
                }))

            cells.append(html.Td(
                html.Div(bar_children, title=tooltip, style={
                    "position":"relative","width":"100%","height":"14px",
                    "background":"#f5f5f5","borderRadius":"2px",
                    "border":f"1px {border_css}",
                }),
                style={"padding":"6px 8px","minWidth":"90px"},
            ))

        rows.append(html.Tr(cells, style={
            "background":row_bg,"opacity":row_opacity,
            "borderBottom":"1px solid #f5f5f5",
        }))

    return html.Table([
        html.Thead(html.Tr(header_cells)),
        html.Tbody(rows),
    ], style={"borderCollapse":"collapse","width":"100%","fontSize":"13px"})

# ── Callback 12 : Classement probabiliste ────────────────────────────────────
# Déclenché quand la sélection ou les poids changent
# → recalcule TOPSIS avec les nouveaux poids et affiche le tableau de classement
@callback(
    Output("recommendation-text","children"),
    *[Input(f"w-{c}","value") for c in CRITERIA],  # un Input par champ de poids
)
def render_recommendation(*weight_vals):
    # Récupérer les poids saisis par l'utilisateur (ou 1.0 si champ vide)
    raw_w = np.array([w if w is not None else 1.0 for w in weight_vals])
    total = raw_w.sum()
    if total == 0:
        total = 1
    w = raw_w / total  # normaliser pour que la somme = 1

    # Recalculer TOPSIS vectorisé sur toutes les simulations (pas de boucle Python)
    # mc_matrices shape = (N_SIM, N_ALT, N_CRIT)
    # Étape 1 : normalisation vectorielle par simulation
    denom = np.sqrt((mc_matrices**2).sum(axis=1, keepdims=True))  # (N_SIM, 1, N_CRIT)
    denom[denom == 0] = 1
    r = mc_matrices / denom  # (N_SIM, N_ALT, N_CRIT)
    # Étape 2 : pondération
    v = r * w  # broadcast (N_SIM, N_ALT, N_CRIT) * (N_CRIT,)
    # Étape 3 : idéal et anti-idéal
    benefit_mask = np.array([c in BENEFIT for c in CRITERIA])  # (N_CRIT,)
    cost_mask = ~benefit_mask
    ideal = np.where(benefit_mask, v.max(axis=1, keepdims=True),
                                   v.min(axis=1, keepdims=True))  # (N_SIM, 1, N_CRIT)
    anti  = np.where(benefit_mask, v.min(axis=1, keepdims=True),
                                   v.max(axis=1, keepdims=True))
    # Étape 4 : distances
    d_plus  = np.sqrt(((v - ideal)**2).sum(axis=2))  # (N_SIM, N_ALT)
    d_minus = np.sqrt(((v - anti)**2).sum(axis=2))
    denom_s = d_plus + d_minus
    denom_s[denom_s == 0] = 1
    scores_w = d_minus / denom_s  # (N_SIM, N_ALT)
    # Étape 5 : rangs
    ranks_w = (N_ALT - scores_w.argsort(axis=1).argsort(axis=1)).astype(int)

    # Calculer les probabilités de rang
    rp = np.zeros((N_ALT, N_ALT))
    for i in range(N_ALT):
        for r in range(1, N_ALT+1):
            rp[i, r-1] = (ranks_w[:, i] == r).mean()

    mean_scores = scores_w.mean(axis=0)  # score moyen par alternative
    best_idx = mean_scores.argmax()       # index de la meilleure alternative
    best_alt = ALTS[best_idx]             # nom de la meilleure alternative

    # Trier par score moyen décroissant (meilleur en premier)
    order = np.argsort(-mean_scores)

    # Construire les lignes du tableau de classement
    rows = []
    for rank_display, idx in enumerate(order, 1):  # rank_display commence à 1
        alt = ALTS[idx]
        sc_mean = mean_scores[idx]
        sc_std  = scores_w[:, idx].std()
        sc_min  = scores_w[:, idx].min()   # pire score obtenu sur les 100k simulations
        sc_max  = scores_w[:, idx].max()   # meilleur score obtenu sur les 100k simulations

        # Échelle commune à toutes les barres : le meilleur score individuel toutes
        # simulations confondues (pour que min/max de chaque alternative rentrent).
        scale = scores_w.max() or 1
        bar_w = int((sc_mean / scale) * 200)  # largeur de la barre pleine (score moyen)
        min_w = int((sc_min  / scale) * 200)  # position de départ de la plage min-max
        max_w = int((sc_max  / scale) * 200)  # position de fin de la plage min-max

        rows.append(html.Tr([
            html.Td(f"#{rank_display}",
                    style={"fontWeight":"bold","width":"55px",
                           "color":"#4472C4" if rank_display==1 else "#666"}),
            html.Td(alt, style={"fontWeight":"bold" if rank_display==1 else "normal",
                                "width":"40px","color":ALT_COLOR[alt]}),
            html.Td(f"{sc_mean:.4f} ± {sc_std:.4f}",
                    style={"fontFamily":"monospace","fontSize":"12px","width":"140px"}),
            # Barre visuelle : plage min → max (couleur pleine) + repère de la moyenne
            html.Td([
                html.Div([
                    # Piste de fond (échelle complète, pour situer la barre)
                    html.Div(style={
                        "position":"absolute","top":"0","bottom":"0","left":"0","width":"200px",
                        "background":"#f0f0f0","borderRadius":"3px",
                    }),
                    # Plage min → max de l'alternative
                    html.Div(style={
                        "position":"absolute","top":"0","bottom":"0",
                        "left":f"{min_w}px","width":f"{max(2, max_w - min_w)}px",
                        "background":rgba(ALT_COLOR[alt], 0.45),
                        "borderRadius":"3px",
                    }),
                    # Repère de la moyenne (petit trait vertical plus foncé)
                    html.Div(style={
                        "position":"absolute","top":"-2px","bottom":"-2px",
                        "left":f"{bar_w}px","width":"3px",
                        "background":ALT_COLOR[alt],
                        "borderRadius":"1px",
                    }),
                ], style={"position":"relative","height":"10px","width":"200px"}),
                # Valeurs min/max toujours visibles (pas seulement au survol)
                html.Div(f"min {sc_min:.3f} · max {sc_max:.3f}",
                          style={"fontSize":"10px","color":"#999","marginTop":"3px"}),
            ], style={"minWidth":"210px"}),
        ], style={"borderBottom":"1px solid #f5f5f5",
                  "background":"#fafcff" if rank_display==1 else "transparent"}))

    # Construire et retourner l'affichage complet (bannière + tableau)
    return html.Div([
        # Bannière : meilleure alternative mise en évidence
        html.Div([
            html.Span("Meilleure alternative : ", style={"fontSize":"14px"}),
            html.Strong(best_alt, style={"fontSize":"22px","color":ALT_COLOR[best_alt]}),
            html.Span(f"  (Score TOPSIS moyen : {mean_scores[best_idx]:.4f} "
                      f"± {scores_w[:,best_idx].std():.4f})",
                      style={"fontSize":"13px","color":"#555","marginLeft":"8px"}),
            html.Br(),
            html.Span(f"P(rang 1) = {rp[best_idx,0]:.1%}  |  "
                      f"P(rang ≤ 3) = {rp[best_idx,:3].sum():.1%}",
                      style={"fontSize":"12px","color":"#888"}),
        ], style={"borderLeft":f"4px solid {ALT_COLOR[best_alt]}",
                  "borderRadius":"6px","padding":"12px 16px","marginBottom":"16px",
                  "background":"#fafcff"}),

        # Tableau complet de classement
        html.Table([
            html.Thead(html.Tr([
                html.Th("Rang"), html.Th("Alt."), html.Th("Score (µ ± σ)"),
                html.Th(""),
            ], style={"borderBottom":"2px solid #ddd","textAlign":"left",
                       "fontSize":"12px","color":"#888"})),
            html.Tbody(rows),
        ], style={"borderCollapse":"collapse","width":"100%","fontSize":"13px"}),
    ])

# ── Callback 13 : Heatmap P(rang) ────────────────────────────────────────────
# Même logique TOPSIS pondérée que render_recommendation (Callback 12), mais
# affiche la probabilité de chaque rang sous forme de heatmap plutôt qu'en texte.
@callback(
    Output("heatmap-rang","figure"),
    *[Input(f"w-{c}","value") for c in CRITERIA],
)
def render_heatmap_rang(*weight_vals):
    raw_w = np.array([w if w is not None else 1.0 for w in weight_vals])
    total = raw_w.sum()
    if total == 0:
        total = 1
    w = raw_w / total

    # Recalculer TOPSIS vectorisé sur toutes les simulations (identique à Callback 12)
    denom = np.sqrt((mc_matrices**2).sum(axis=1, keepdims=True))
    denom[denom == 0] = 1
    r = mc_matrices / denom
    v = r * w
    benefit_mask = np.array([c in BENEFIT for c in CRITERIA])
    ideal = np.where(benefit_mask, v.max(axis=1, keepdims=True),
                                   v.min(axis=1, keepdims=True))
    anti  = np.where(benefit_mask, v.min(axis=1, keepdims=True),
                                   v.max(axis=1, keepdims=True))
    d_plus  = np.sqrt(((v - ideal)**2).sum(axis=2))
    d_minus = np.sqrt(((v - anti)**2).sum(axis=2))
    denom_s = d_plus + d_minus
    denom_s[denom_s == 0] = 1
    scores_w = d_minus / denom_s
    ranks_w = (N_ALT - scores_w.argsort(axis=1).argsort(axis=1)).astype(int)

    # rp[i, r] = P(alternative i est classée rang r+1)
    rp = np.zeros((N_ALT, N_ALT))
    for i in range(N_ALT):
        for rnk in range(1, N_ALT + 1):
            rp[i, rnk-1] = (ranks_w[:, i] == rnk).mean()

    mean_scores = scores_w.mean(axis=0)
    order = np.argsort(-mean_scores)  # tri par score moyen décroissant

    z = (rp[order] * 100).tolist()
    y = [ALTS[idx] for idx in order]
    x = [f"#{r}" for r in range(1, N_ALT + 1)]
    text = [[f"{val:.0f}%" for val in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y, colorscale="Blues",
        text=text, texttemplate="%{text}",
        hovertemplate="%{y} — %{x} : %{z:.1f}%<extra></extra>",
        colorbar=dict(title="P(%)"),
    ))
    fig.update_layout(
        title="P(rang) — Probabilité de chaque rang",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=20, t=40, b=40),
        yaxis=dict(autorange="reversed"),  # meilleure alternative en haut
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# LANCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # debug=False obligatoire avec Python 3.14 (debug=True cause une erreur)
    # port=8051 → accéder via http://127.0.0.1:8051
    app.run(debug=False, port=8051, threaded=True)
