import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

CSV_FILENAME = "season-1819-PL.csv"  # put file in same folder

def points_for_result(result, side):
    if result == 'D':
        return 1
    if side == 'home' and result == 'H':
        return 3
    if side == 'away' and result == 'A':
        return 3
    return 0

def compute_recent_stats(df, n=5):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y", dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    history = {}
    cols = [
        'home_pts_last5_avg','away_pts_last5_avg',
        'home_gd_last5_avg','away_gd_last5_avg',
        'home_gf_last5_avg','away_gf_last5_avg'
    ]
    for c in cols:
        df[c] = np.nan
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        result = row['FTR']
        hg = int(row['FTHG']) if not pd.isna(row['FTHG']) else 0
        ag = int(row['FTAG']) if not pd.isna(row['FTAG']) else 0
        def summarize(team):
            rec = history.get(team, [])
            lastn = rec[-n:]
            if not lastn:
                return {'pts_avg': np.nan, 'gd_avg': np.nan, 'gf_avg': np.nan}
            pts = [r['points'] for r in lastn]
            gd = [r['gd'] for r in lastn]
            gf = [r['gf'] for r in lastn]
            return {'pts_avg': np.mean(pts), 'gd_avg': np.mean(gd), 'gf_avg': np.mean(gf)}
        home_stats = summarize(home)
        away_stats = summarize(away)
        df.at[idx, 'home_pts_last5_avg'] = home_stats['pts_avg']
        df.at[idx, 'away_pts_last5_avg'] = away_stats['pts_avg']
        df.at[idx, 'home_gd_last5_avg'] = home_stats['gd_avg']
        df.at[idx, 'away_gd_last5_avg'] = away_stats['gd_avg']
        df.at[idx, 'home_gf_last5_avg'] = home_stats['gf_avg']
        df.at[idx, 'away_gf_last5_avg'] = away_stats['gf_avg']
        home_pts = points_for_result(result, 'home')
        away_pts = points_for_result(result, 'away')
        history.setdefault(home, []).append({'points': home_pts, 'gf': hg, 'ga': ag, 'gd': hg-ag, 'date': row['Date']})
        history.setdefault(away, []).append({'points': away_pts, 'gf': ag, 'ga': hg, 'gd': ag-hg, 'date': row['Date']})
    df_features = df.dropna(subset=['home_pts_last5_avg','away_pts_last5_avg']).reset_index(drop=True)
    return df_features

def build_and_train(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for c in ['HomeTeam','AwayTeam','FTHG','FTAG','FTR','Date']:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")
    df_feat = compute_recent_stats(df, n=5)
    features = [
        'home_pts_last5_avg','away_pts_last5_avg',
        'home_gd_last5_avg','away_gd_last5_avg',
        'home_gf_last5_avg','away_gf_last5_avg'
    ]
    for odds_col in ['B365H','B365D','B365A']:
        if odds_col in df_feat.columns:
            features.append(odds_col)
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df_feat['HomeEnc'] = le_home.fit_transform(df_feat['HomeTeam'])
    df_feat['AwayEnc'] = le_away.fit_transform(df_feat['AwayTeam'])
    features += ['HomeEnc','AwayEnc']
    X = df_feat[features].fillna(0).values
    y = df_feat['FTR'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    artifacts = {
        'model': model,
        'le_home': le_home,
        'le_away': le_away,
        'features': features,
        'df_full': df,
        'df_feat': df_feat,
    }
    return artifacts, acc

def latest_team_stats_from_df(df_raw, team, n=5):
    tmp = df_raw.copy()
    tmp['Date'] = pd.to_datetime(tmp['Date'], format="%d/%m/%Y", dayfirst=True, errors='coerce')
    tmp = tmp.sort_values('Date')
    is_home = tmp['HomeTeam'] == team
    is_away = tmp['AwayTeam'] == team
    team_matches = tmp[is_home | is_away].copy()
    if team_matches.empty:
        return None
    last_matches = team_matches.tail(n)
    pts = []
    gfs = []
    gds = []
    for _, r in last_matches.iterrows():
        if r['HomeTeam'] == team:
            gf = int(r['FTHG']); ga = int(r['FTAG'])
            result = r['FTR']
            pts.append(points_for_result(result, 'home'))
            gfs.append(gf); gds.append(gf - ga)
        else:
            gf = int(r['FTAG']); ga = int(r['FTHG'])
            result = r['FTR']
            pts.append(points_for_result(result, 'away'))
            gfs.append(gf); gds.append(gf - ga)
    return {
        'pts_avg': np.mean(pts) if pts else 0,
        'gd_avg': np.mean(gds) if gds else 0,
        'gf_avg': np.mean(gfs) if gfs else 0
    }
