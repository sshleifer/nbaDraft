import pandas as pd
from assemble_dataset import order

def clean_colpro(colpro):
    return order(colpro[column_order], column_order)

column_order = [
    'Name', u'Team','pick', 'age', 'pos', u'GP', u'Min',
    u'Pts', u'FG', u'FGA', u'FG%', u'2Pt', u'2PtA', u'2P%', u'3Pt',
    u'3PtA', u'3P%', u'FTM', u'FTA', u'FT%', u'Off', u'Def', u'TOT',
    u'Asts', u'Stls', u'Blks', u'TOs', u'PFs', u'year', u'PTs/g',
    u'FGA/g',u'Pts/Play', u'TS%', u'eFG%', u'FTA/FGA', u'3PA/FGA',
    u'Ast/g',u'Ast/FGA', u'A/TO', u'PPR', u'BK/g', u'STL/g', u'PF/g',
    u'Weight', u'Reach', u'Body Fat',u'Hand Length', u'Hand Width',
    u'No Step Vert', u'Max Vert', u'Bench', u'Agility', u'Sprint',
    u'Drafted',  u'heightshoes', u'heightbare', u'wingspan', u'standvertreach', u'maxvertreach',
]
