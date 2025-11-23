import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def predict_result(home, away, home_df, away_df, model):
    home_stats = home_df[home_df['Home'] == home]
    away_stats = away_df[away_df['Away'] == away]
    home_stats = home_stats.reset_index(drop=True)
    away_stats = away_stats.reset_index(drop=True)

    form_diff = home_stats['Home_Recent_Form'] - away_stats['Away_Recent_Form']
    
    final_df = pd.concat([home_stats, away_stats], axis=1)
    final_df['Form_Diff'] = form_diff
    final_df['Match_Period_Morning'] = False
    final_df['Match_Period_Afternoon'] = True
    final_df['Match_Period_Evening'] = False

    model_features = [
        'Average_Score_gols_H', 'Average_Score_gols_A',
        'Average_Conceding_gols_H', 'Average_Conceding_gols_A', 'Average_xG_H',
        'Average_xG_A', 'Average_ShotsTotal_H', 'Average_ShotsTotal_A',
        'Average_ShotsTarget_H', 'Average_ShotsTarget_A',
        'Average_Possession_H', 'Average_Possession_A', 'Average_PassTotal_H',
        'Average_SaveTotal_H', 'Average_SaveTotal_A', 'Average_SaveCompleted_H',
        'Average_SaveCompleted_A', 'Average_Corners_H', 'Average_Corners_A',
        'Home_General_Avg_Score_gols', 'Home_General_Avg_Conceding_gols',
        'Home_General_Avg_xG', 'Home_General_Avg_ShotsTotal',
        'Home_General_Avg_ShotsTarget', 'Home_General_Avg_Possession',
        'Away_General_Avg_Score_gols', 'Away_General_Avg_Conceding_gols',
        'Away_General_Avg_xG', 'Away_General_Avg_ShotsTotal',
        'Away_General_Avg_ShotsTarget', 'Away_General_Avg_Possession',
        'Home_Recent_Form', 'Away_Recent_Form', 'Form_Diff',
        'Match_Period_Afternoon', 'Match_Period_Evening',
        'Match_Period_Morning'
    ]

    final_df = final_df[model_features]

    return(model.predict(final_df))

    

